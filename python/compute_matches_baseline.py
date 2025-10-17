#!/usr/bin/env python3
"""
Baseline CNN MobileNetV2 Pet Matching (Standalone)
==================================================

This script implements the Baseline Siamese CNN (MobileNetV2) matching flow
with the same caching infrastructure used by the Proposed model.

It provides:
- Thread-safe JSON caching for embeddings and thumbnails
- File-based thumbnail storage (JSON stores paths, not base64)
- Model-aware cache invalidation
- Consistent JSON output format for PHP integration

Usage:
    python compute_matches_baseline.py <query_image> <pet_types> <gallery_dir> <top_k> [--debug]
"""

import os
import sys
import json
import time
import base64
import hashlib
import threading
import atexit
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

# Deep learning imports for the standalone Baseline model
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

# ============================================================================
# STANDALONE BASELINE PET MATCHER (inlined)
# ============================================================================

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)


class PetMatcher:
    """
    Pet matching system using trained Siamese CNN MobileNetV2 model.
    Loads the model and computes similarity matches between pet images.
    """

    def __init__(self, model_path='../model/best_model.h5', input_shape=(224, 224, 3)):
        self.model_path = model_path
        self.input_shape = input_shape
        self.model = None
        self.base_network = None
        self._load_model()

    def euclidean_distance(self, vectors):
        x, y = vectors
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def contrastive_loss(self, y_true, y_pred, margin=1.0):
        y_true = K.cast(y_true, K.floatx())
        y_pred = K.cast(y_pred, K.floatx())
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def siamese_accuracy(self, y_true, y_pred):
        # Map distance to similarity then threshold dynamically
        y_pred_similarity = 0.5 * (1.0 + tf.tanh(1.0 - y_pred))
        dynamic_threshold = tf.reduce_mean(y_pred_similarity)
        dynamic_threshold = tf.clip_by_value(dynamic_threshold, 0.3, 0.7)
        predictions = K.cast(y_pred_similarity > dynamic_threshold, K.floatx())
        y_true_cast = K.cast(y_true, K.floatx())
        correct_predictions = K.cast(K.equal(y_true_cast, predictions), K.floatx())
        acc = K.mean(correct_predictions)
        acc = tf.where(tf.math.is_finite(acc), acc, 0.5)
        return K.clip(acc, 0.0, 1.0)

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Some Lambda layers may reference K from builtins
        import builtins
        if not hasattr(builtins, 'K'):
            builtins.K = K

        def l2_normalize(x):
            return K.l2_normalize(x, axis=1)

        def euclidean_distance_lambda(vectors):
            return self.euclidean_distance(vectors)

        custom_objects = {
            'contrastive_loss': lambda y_true, y_pred: self.contrastive_loss(y_true, y_pred, margin=1.0),
            'siamese_accuracy': self.siamese_accuracy,
            'euclidean_distance': self.euclidean_distance,
            'l2_normalize': l2_normalize,
            '<lambda>': lambda y_true, y_pred: self.contrastive_loss(y_true, y_pred, margin=1.0),
            '<lambda_1>': l2_normalize,
            '<lambda_2>': euclidean_distance_lambda,
            'K': K,
        }

        # Load siamese model
        self.model = load_model(self.model_path, custom_objects=custom_objects, compile=False)
        # Compile for completeness (not used for inference logic)
        try:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                loss=lambda y_true, y_pred: self.contrastive_loss(y_true, y_pred, margin=1.0),
                metrics=[self.siamese_accuracy]
            )
        except Exception:
            pass

        # Try to extract base network (embedding extractor)
        self._extract_base_network()

    def _extract_base_network(self):
        try:
            for layer in self.model.layers:
                if 'MobileNetV2_Base' in layer.name or isinstance(layer, tf.keras.Model):
                    if hasattr(layer, 'output') and len(layer.output.shape) == 2:
                        self.base_network = layer
                        break
            if self.base_network is None:
                # Fallback: find l2_normalize/embedding layer
                for layer in self.model.layers:
                    if 'l2_normalize' in layer.name or 'embeddings' in layer.name:
                        input_tensor = self.model.input[0] if isinstance(self.model.input, list) else self.model.input
                        self.base_network = tf.keras.Model(inputs=input_tensor, outputs=layer.output,
                                                           name='extracted_base_network')
                        break
        except Exception:
            self.base_network = None

    def load_and_preprocess_image(self, image_path):
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((self.input_shape[0], self.input_shape[1]), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            if img_array.shape != self.input_shape:
                raise ValueError(f"Image shape mismatch: {img_array.shape} vs {self.input_shape}")
            return img_array
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {str(e)}")

    def get_embedding(self, image_path):
        img_array = self.load_and_preprocess_image(image_path)
        img_batch = np.expand_dims(img_array, axis=0)
        if self.base_network is not None:
            embedding = self.base_network.predict(img_batch, verbose=0)
        else:
            input_tensor = self.model.input[0] if isinstance(self.model.input, list) else self.model.input
            # Find embedding layer by name
            embedding_layer = None
            for layer in self.model.layers:
                if 'l2_normalize' in layer.name:
                    embedding_layer = layer
                    break
            if embedding_layer is None:
                raise ValueError('Could not find embedding layer in model')
            temp_model = tf.keras.Model(inputs=input_tensor, outputs=embedding_layer.output)
            embedding = temp_model.predict(img_batch, verbose=0)
        return embedding[0]

    def batch_compute_embeddings(self, image_paths, batch_size=32):
        if len(image_paths) == 0:
            return np.array([])
        embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            for p in batch_paths:
                try:
                    arr = self.load_and_preprocess_image(p)
                    batch_images.append(arr)
                except Exception:
                    batch_images.append(np.zeros(self.input_shape, dtype=np.float32))
            batch_array = np.stack(batch_images, axis=0)
            if self.base_network is not None:
                batch_embeddings = self.base_network.predict(batch_array, verbose=0)
            else:
                input_tensor = self.model.input[0] if isinstance(self.model.input, list) else self.model.input
                embedding_layer = None
                for layer in self.model.layers:
                    if 'l2_normalize' in layer.name:
                        embedding_layer = layer
                        break
                if embedding_layer is None:
                    raise ValueError('Could not find embedding layer in model')
                tmp_model = tf.keras.Model(inputs=input_tensor, outputs=embedding_layer.output)
                batch_embeddings = tmp_model.predict(batch_array, verbose=0)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings) if embeddings else np.array([])

    def compute_similarity(self, embedding1, embedding2):
        """Distance-to-similarity mapping with exponential decay.
        Ensures identical images (distance=0) map to similarity=1.0.
        """
        distance = float(np.linalg.norm(embedding1 - embedding2))
        return float(np.exp(-distance))

# ============================================================================
# CACHE MANAGER (Embeddings + Thumbnails) - same structure as CapsNet
# ============================================================================

class _JSONCache:
    """Thread-safe JSON cache with lazy writes."""
    def __init__(self, path: Path, root_key: str):
        self.path = Path(path)
        self.root_key = root_key
        self._lock = threading.RLock()
        self._dirty = False
        self._data = {
            "version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            self.root_key: {}
        }
        try:
            if self.path.exists():
                with self.path.open('r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and self.root_key in loaded:
                        self._data = loaded
        except Exception:
            pass

    def get(self, key):
        with self._lock:
            return self._data.get(self.root_key, {}).get(key)

    def set(self, key, value):
        with self._lock:
            self._data.setdefault(self.root_key, {})[key] = value
            self._dirty = True

    def flush(self):
        with self._lock:
            if not self._dirty:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open('w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False)
            tmp.replace(self.path)
            self._dirty = False


class CacheManager:
    """Coordinates caches for embeddings and thumbnails (Baseline model)."""
    def __init__(self):
        # Ensure absolute path to project root (fin)
        root = Path(__file__).resolve().parent.parent  # go up from python/ to fin/
        self.cache_dir = root / 'cache' / 'Baseline'
        self.thumbs_dir = self.cache_dir / 'thumbs'
        self.emb_path = self.cache_dir / 'embeddings_cache.json'
        self.thm_path = self.cache_dir / 'thumbnails_cache.json'
        self.emb_cache = _JSONCache(self.emb_path, 'embeddings')
        self.thm_cache = _JSONCache(self.thm_path, 'thumbnails')
        atexit.register(self.flush)

    @staticmethod
    def _file_fingerprint(path: Path) -> str:
        try:
            stat = path.stat()
            payload = f"{path.resolve()}|{int(stat.st_mtime)}|{stat.st_size}".encode('utf-8', 'ignore')
            return hashlib.sha1(payload).hexdigest()
        except Exception:
            return hashlib.sha1(str(path).encode('utf-8', 'ignore')).hexdigest()

    @staticmethod
    def _model_namespace(model_path: str) -> str:
        """Model fingerprint for cache invalidation."""
        if model_path:
            p = Path(model_path)
            try:
                stat = p.stat()
                return f"baseline|{p.name}|{int(stat.st_mtime)}|{stat.st_size}"
            except Exception:
                return f"baseline|{p.name}"
        return 'baseline|model'

    def embedding_key(self, model_path: str, img_path: str) -> str:
        p = Path(img_path)
        return f"emb|{self._model_namespace(model_path)}|{self._file_fingerprint(p)}"

    def thumbnail_key(self, img_path: str, size) -> str:
        p = Path(img_path)
        return f"thm|{size[0]}x{size[1]}|{self._file_fingerprint(p)}"

    def get_embedding(self, key: str):
        return self.emb_cache.get(key)

    def set_embedding(self, key: str, vec: np.ndarray):
        try:
            self.emb_cache.set(key, {
                'updated': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                'vector': vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
            })
        except Exception:
            pass

    def get_thumbnail(self, key: str):
        return self.thm_cache.get(key)

    def _thumb_path_for_key(self, key: str, ext: str = '.jpg') -> Path:
        h = hashlib.sha1(key.encode('utf-8', 'ignore')).hexdigest()
        subdir = h[:2]
        return self.thumbs_dir / subdir / f"{h}{ext}"

    def set_thumbnail_file(self, key: str, data_bytes: bytes, mime: str = 'image/jpeg'):
        try:
            self.thumbs_dir.mkdir(parents=True, exist_ok=True)
            dst = self._thumb_path_for_key(key, '.jpg')
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, 'wb') as f:
                f.write(data_bytes)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            rel = os.path.relpath(dst, self.cache_dir)
            self.thm_cache.set(key, {
                'updated': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                'mime': mime,
                'file': rel.replace('\\', '/')
            })
        except Exception:
            pass

    def set_thumbnail(self, key: str, b64: str, mime: str = 'image/jpeg'):
        try:
            data_bytes = base64.b64decode(b64.encode('utf-8'))
            self.set_thumbnail_file(key, data_bytes, mime=mime)
        except Exception:
            pass

    def flush(self):
        try:
            self.emb_cache.flush()
            self.thm_cache.flush()
        except Exception:
            pass


# Singleton cache manager for Baseline model
_CACHE = CacheManager()
try:
    _CACHE.cache_dir.mkdir(parents=True, exist_ok=True)
    _CACHE.thumbs_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# ============================================================================
# BASELINE ENGINE WRAPPER WITH CACHING
# ============================================================================

class BaselineEngineWrapper:
    """Wrapper around PetMatcher with caching infrastructure."""
    
    def __init__(self, model_path, debug=False, batch_size=32, num_workers=None):
        self.model_path = model_path
        self.debug = debug
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers else cpu_count()
        self.matcher = PetMatcher(model_path=model_path)
        self.debug_info = []
        self.log(f"Baseline model loaded from {model_path}")
        self.log(f"Using {self.num_workers} workers with batch size {self.batch_size}")

    def log(self, msg):
        if self.debug:
            self.debug_info.append(msg)
            print(f"[DEBUG] {msg}", file=sys.stderr)

    def get_embedding(self, img_path):
        """Get embedding with cache support."""
        # Try cache first
        try:
            key = _CACHE.embedding_key(self.model_path, img_path)
            cached = _CACHE.get_embedding(key)
            if cached and 'vector' in cached:
                vec = np.array(cached['vector'], dtype=np.float32)
                if vec.size > 0:
                    return vec
        except Exception:
            pass

        # Compute embedding
        try:
            embedding = self.matcher.get_embedding(img_path)
            # Cache it
            try:
                key = _CACHE.embedding_key(self.model_path, img_path)
                _CACHE.set_embedding(key, embedding)
            except Exception:
                pass
            return embedding
        except Exception as e:
            self.log(f"Error getting embedding for {img_path}: {e}")
            raise

    def get_embeddings_batch(self, img_paths):
        """Batch embedding extraction with cache support."""
        if len(img_paths) == 0:
            return np.array([]), []

        outputs = [None] * len(img_paths)
        valid_paths = []
        to_process = []
        uncached_paths = []

        # Check cache
        for idx, p in enumerate(img_paths):
            try:
                key = _CACHE.embedding_key(self.model_path, p)
                cached = _CACHE.get_embedding(key)
                if cached and 'vector' in cached:
                    vec = np.array(cached['vector'], dtype=np.float32)
                    outputs[idx] = vec
                    valid_paths.append(p)
                else:
                    to_process.append(idx)
                    uncached_paths.append(p)
            except Exception:
                to_process.append(idx)
                uncached_paths.append(p)

        if len(uncached_paths) == 0:
            final_vecs = [outputs[i] for i, p in enumerate(img_paths) if outputs[i] is not None]
            final_paths = [p for i, p in enumerate(img_paths) if outputs[i] is not None]
            return np.vstack(final_vecs) if final_vecs else np.array([]), final_paths

        # Compute uncached using baseline matcher's batch method
        try:
            embeddings = self.matcher.batch_compute_embeddings(uncached_paths, batch_size=self.batch_size)
            for j, idx in enumerate(to_process):
                if j < len(embeddings):
                    vec = embeddings[j]
                    outputs[idx] = vec
                    valid_paths.append(uncached_paths[j])
                    # Cache it
                    try:
                        key = _CACHE.embedding_key(self.model_path, uncached_paths[j])
                        _CACHE.set_embedding(key, vec)
                    except Exception:
                        pass
        except Exception as e:
            self.log(f"Batch processing failed: {e}")
            # Fallback: individual processing
            for i, p in zip(to_process, uncached_paths):
                try:
                    vec = self.get_embedding(p)
                    outputs[i] = vec
                    valid_paths.append(p)
                except Exception as e2:
                    self.log(f"Failed to process {p}: {e2}")

        final_vecs = [outputs[i] for i, p in enumerate(img_paths) if outputs[i] is not None]
        final_paths = [p for i, p in enumerate(img_paths) if outputs[i] is not None]
        return np.vstack(final_vecs) if final_vecs else np.array([]), final_paths

    def compute_similarity(self, emb1, emb2):
        """Compute similarity between embeddings (cosine similarity)."""
        return self.matcher.compute_similarity(emb1, emb2)

    def generate_thumbnail(self, img_path, size=(300, 300)):
        """Generate thumbnail with file-based caching."""
        # Normalize path FIRST for consistent cache keys
        img_path = str(Path(img_path).resolve())
        key = _CACHE.thumbnail_key(img_path, size)
        
        # Try cache first
        try:
            cached = _CACHE.get_thumbnail(key)
            if cached and isinstance(cached, dict):
                if 'file' in cached:
                    thumb_file = Path(_CACHE.cache_dir) / cached['file']
                    if thumb_file.exists():
                        self.log(f"Using cached thumbnail for {img_path}")
                        with open(thumb_file, 'rb') as f:
                            return base64.b64encode(f.read()).decode('utf-8')
                if 'data_base64' in cached:
                    return cached['data_base64']
        except Exception as e:
            self.log(f"Cache read error for {img_path}: {str(e)}")

        # Generate new thumbnail
        try:
            if not Path(img_path).exists():
                self.log(f"ERROR: Image file not found: {img_path}")
                return None
                
            img = Image.open(img_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            data = buffer.read()
            try:
                _CACHE.set_thumbnail_file(key, data, mime='image/jpeg')
                self.log(f"Cached thumbnail for {img_path}: {key}")
            except Exception as e:
                self.log(f"ERROR: Failed to cache thumbnail for {img_path}: {str(e)}")
            return base64.b64encode(data).decode('utf-8')
        except Exception as e:
            self.log(f"Error generating thumbnail for {img_path}: {e}")
            return None

    def generate_thumbnails_parallel(self, img_paths, size=(150, 150)):
        """Generate thumbnails in parallel."""
        thumbnails = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {executor.submit(self.generate_thumbnail, path, size): path 
                             for path in img_paths}
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    thumb = future.result()
                    thumbnails[path] = thumb
                except Exception as e:
                    self.log(f"Thumbnail generation failed for {path}: {e}")
                    thumbnails[path] = None
        return thumbnails

    def find_matches(self, query_image_path, gallery_dir, pet_types, top_k=10):
        """Find top-k matches using baseline model."""
        self.log(f"Finding matches for: {query_image_path}")
        self.log(f"Gallery directory: {gallery_dir}")
        self.log(f"Pet types: {pet_types}")
        self.log(f"Top K: {top_k}")

        pet_types_list = [pt.strip().title() for pt in pet_types.split(',')]
        if 'Unknown' in pet_types_list:
            pet_types_list = ['Unknown']
        self.log(f"Parsed pet types: {pet_types_list}")

        # Get query embedding
        try:
            query_embedding = self.get_embedding(query_image_path)
            self.log(f"Query embedding shape: {query_embedding.shape}")
        except Exception as e:
            raise Exception(f"Failed to extract query embedding: {str(e)}")

        # Collect gallery images
        gallery_images = []
        for pet_type in pet_types_list:
            dir_name = 'Unknown' if pet_type == 'Unknown' else pet_type + 's'
            type_dir = os.path.join(gallery_dir, dir_name)
            if not os.path.exists(type_dir):
                self.log(f"Warning: Directory not found: {type_dir}")
                continue
            self.log(f"Scanning directory: {type_dir}")
            for filename in os.listdir(type_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    img_path = str(Path(os.path.join(type_dir, filename)).resolve())
                    gallery_images.append({
                        'path': img_path,
                        'filename': filename,
                        'type': pet_type
                    })

        self.log(f"Found {len(gallery_images)} gallery images")
        if len(gallery_images) == 0:
            raise Exception(f"No gallery images found in {gallery_dir}")

        # Process gallery images in batches
        gallery_paths = [item['path'] for item in gallery_images]
        self.log(f"Processing gallery images in batches of {self.batch_size}...")

        gallery_embeddings_list = []
        for i in range(0, len(gallery_paths), self.batch_size):
            batch_paths = gallery_paths[i:i + self.batch_size]
            try:
                batch_embeddings, valid_paths = self.get_embeddings_batch(batch_paths)
                gallery_embeddings_list.extend(batch_embeddings)
                self.log(f"Processed batch {i // self.batch_size + 1}/{(len(gallery_paths) + self.batch_size - 1) // self.batch_size}")
            except Exception as e:
                self.log(f"Error processing batch starting at index {i}: {str(e)}")

        if len(gallery_embeddings_list) == 0:
            raise Exception("Failed to extract any gallery embeddings")

        self.log(f"Successfully extracted {len(gallery_embeddings_list)} embeddings")

        # Compute similarities using improved Euclidean distance normalization
        # This allows identical images to reach 100% similarity
        gallery_embeddings = np.array(gallery_embeddings_list)
        similarities = np.zeros(len(gallery_embeddings))
        for i, gallery_emb in enumerate(gallery_embeddings):
            distance = np.linalg.norm(query_embedding - gallery_emb)
            # Improved formula: identical images (distance=0) → similarity=1.0 (100%)
            # Using exponential decay for better distribution
            similarities[i] = np.exp(-distance)

        # Create matches list
        matches = []
        for i, (gallery_item, similarity) in enumerate(zip(gallery_images[:len(similarities)], similarities)):
            matches.append({
                'path': gallery_item['path'],
                'filename': gallery_item['filename'],
                'type': gallery_item['type'],
                'similarity': float(similarity),
                'distance': float(1.0 - similarity)
            })

        self.log(f"Successfully computed {len(matches)} similarities")

        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        top_k_matches = matches[:top_k]

        # Generate thumbnails
        self.log(f"Generating thumbnails for top {top_k} matches in parallel...")
        top_k_paths = [match['path'] for match in top_k_matches]
        thumbnails = self.generate_thumbnails_parallel(top_k_paths)

        # Build final results
        top_matches = []
        for rank, match in enumerate(top_k_matches, start=1):
            top_matches.append({
                'rank': rank,
                'path': match['path'],
                'filename': match['filename'],
                'type': match['type'],
                'similarity': match['similarity'],
                'distance': match['distance'],
                'thumb_base64': thumbnails.get(match['path'])
            })

        self.log(f"Top match similarity: {top_matches[0]['similarity']:.4f}" if top_matches else "No matches")
        return top_matches


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main CLI interface."""
    if len(sys.argv) < 5:
        print(json.dumps({
            'ok': False,
            'error': 'Usage: compute_matches_baseline.py <query_image> <pet_types> <gallery_dir> <top_k> [--debug]'
        }))
        sys.exit(1)

    query_image = sys.argv[1]
    pet_types = sys.argv[2]
    gallery_dir = sys.argv[3]
    top_k = int(sys.argv[4])
    debug = '--debug' in sys.argv

    # Resolve model path with fallbacks
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    candidate_paths = [
        os.path.join(root_dir, 'model', 'Baseline', 'best_model.h5'),
        os.path.join(root_dir, 'model', 'best_model.h5'),
    ]
    model_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])

    result = {
        'ok': False,
        'matches': [],
        'debug': [],
        'model': 'baseline_cnn'
    }

    try:
        num_workers = cpu_count()
        batch_size = min(32, max(8, num_workers * 2))

        engine = BaselineEngineWrapper(
            model_path,
            debug=debug,
            batch_size=batch_size,
            num_workers=num_workers
        )

        matches = engine.find_matches(query_image, gallery_dir, pet_types, top_k)

        # Format results for PHP
        formatted_matches = []
        for match in matches:
            formatted_matches.append({
                'rank': match['rank'],
                'path': match['path'].replace('\\', '/'),
                'filename': match['filename'],
                'type': match['type'],
                'similarity': round(match['similarity'], 4),
                'distance': round(match['distance'], 4),
                'confidence': round(match['similarity'] * 100, 2),
                'score': round(match['similarity'] * 100, 2),
                'thumb_base64': match['thumb_base64']
            })

        result['ok'] = True
        result['matches'] = formatted_matches
        result['debug'] = engine.debug_info

    except Exception as e:
        result['error'] = str(e)
        result['debug'] = [f"Exception: {str(e)}"]
        if debug:
            import traceback
            result['debug'].append(traceback.format_exc())

    print(json.dumps(result, indent=2 if debug else None))


if __name__ == '__main__':
    main()
