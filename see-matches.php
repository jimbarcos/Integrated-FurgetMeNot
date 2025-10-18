<?php 
set_time_limit(300); // Set execution time limit to 5 minutes
$serverStartMs = microtime(true)*1000.0; 
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>See Matches - Fur-Get Me Not</title>
    <link href="./css/styles.css" rel="stylesheet">
    <link href="./css/components.css" rel="stylesheet">

    <style>
        /* Enhanced card + name styling */
        .top-match-card {
            background:#ffffff;
            border:1px solid #d9e2f2;
            border-radius:20px;
            padding:16px 16px 18px;
            min-width: 260px;
            display:flex;
            flex-direction:column;
            position:relative;
            box-shadow:0 2px 4px rgba(30,60,120,0.06), 0 4px 14px -4px rgba(30,60,120,0.10);
            transition:box-shadow .25s, transform .25s, border-color .25s;
        }
        .top-match-card:hover, .top-match-card:focus-within {
            box-shadow:0 6px 22px -4px rgba(30,60,120,0.25);
            transform:translateY(-3px);
            border-color:#b3c4dd;
        }
        .top-match-card:focus-within { outline:2px solid #5b8bff; outline-offset:2px; }
        .top-match-image-wrapper { aspect-ratio:1/1; background:linear-gradient(135deg,#f3f7fc,#e9f0fa); display:flex; align-items:center; justify-content:center; }
        .top-match-image { width:100%; height:100%; object-fit:cover; }
        .match-name {
            margin-top:8px;
            font-weight:600;
            font-size:0.95rem;
            line-height:1.25;
            color:#223a7b;
            display:-webkit-box;
            -webkit-line-clamp:2;
            -webkit-box-orient:vertical;
            overflow:hidden;
            text-overflow:ellipsis;
            min-height:2.4em; /* reserve space for two lines */
            word-break:break-word;
            cursor:default;
            transition:color .25s;
        }
        .match-name:hover { color:#1b4fc1; }
        .see-image-btn, .details-btn { transition:background .25s, color .25s, box-shadow .25s; }
        .see-image-btn:hover { background:#ffffff; box-shadow:0 2px 10px rgba(56,103,214,0.25); }
        .details-btn:hover { filter:brightness(1.05); }
    </style>
</head>
<body>
    <div class="bg-paws"></div>
    <header>
        <div class="header-bar"></div>
        <a href="index.php" class="back-arrow">
            <img src="assets/How-it-Works/back-arrow.png" alt="Back">
        </a>
        <div class="subpage-header">
            <img src="assets/Logos/interface-setting-app-widget--Streamline-Core.png" alt="Matches Icon" class="subpage-icon">
            <h1 class="subpage-title">See matches</h1>
            <div class="subpage-subpill">Find matches of your pet</div>
        </div>
    </header>
    <main>
        <div class="see-matches-card">
            <div class="card-subtitle-holder">
                <div class="card-subtitle">Your Search Image</div>
                <div id="procTimePill" class="process-time"></div>
            </div>
            <div class="see-matches-row">
                <?php
                $originalBase64 = null; $processedBase64 = null; $petType = 'Unknown'; $confidence = 0; $steps = [];$errorMsg = null; $matches = []; $matchesBaseline = []; $topKDefault = 3; $similarityDebug = []; $similarityAttempts = [];
                $preprocessEnabled = isset($_POST['preprocess']); // checkbox present only if enabled
                $clientStartMs = isset($_POST['client_start_ms']) ? (float)$_POST['client_start_ms'] : 0; // client timestamp when submit clicked
                $userPetType = isset($_POST['pet-type']) ? trim($_POST['pet-type']) : '';
                $typeSelectedByUser = in_array($userPetType, ['Dog','Cat'], true);
                $requestedTop = isset($_POST['top_k']) ? max(1, intval($_POST['top_k'])) : $topKDefault;
                if (isset($_FILES['pet-image']) && $_FILES['pet-image']['error'] === UPLOAD_ERR_OK) {
                    $uploadDir = __DIR__ . DIRECTORY_SEPARATOR . 'uploads';
                    if (!is_dir($uploadDir)) { @mkdir($uploadDir, 0777, true); }
                    $tmp = $_FILES['pet-image']['tmp_name'];
                    $ext = pathinfo($_FILES['pet-image']['name'], PATHINFO_EXTENSION);
                    $safeName = 'img_' . date('Ymd_His') . '_' . bin2hex(random_bytes(4)) . '.' . $ext;
                    $destPath = $uploadDir . DIRECTORY_SEPARATOR . $safeName;
                    @move_uploaded_file($tmp, $destPath);

                    $python = 'python';
                    $processScript = __DIR__ . DIRECTORY_SEPARATOR . 'python' . DIRECTORY_SEPARATOR . 'process_image.py';
                    if ($preprocessEnabled) {
                        if (file_exists($processScript)) {
                            $cmd = escapeshellcmd($python . ' ' . escapeshellarg($processScript) . ' ' . escapeshellarg($destPath));
                            $output = shell_exec($cmd . ' 2>&1');
                            if ($output) {
                                $firstBrace = strpos($output, '{');
                                if ($firstBrace !== false) {
                                    $candidate = substr($output, $firstBrace);
                                    $json = json_decode($candidate, true);
                                } else {
                                    $json = json_decode($output, true);
                                }
                                if (isset($json['ok']) && $json['ok']) {
                                    $originalBase64 = $json['original_base64'] ?? null;
                                    $processedBase64 = $json['processed_base64'] ?? null;
                                    $petType = $json['pet_type'] ?? 'Unknown';
                                    $confidence = $json['confidence'] ?? 0;
                                    if ($typeSelectedByUser) { $petType = $userPetType; }
                                    $steps = $json['steps'] ?? [];
                                } elseif ($json) {
                                    $errorMsg = $json['error'] ?? 'Unknown processing error';
                                } else {
                                    $errorMsg = 'Failed to parse preprocessing JSON';
                                }
                            } else { $errorMsg = 'No output from Python process script'; }
                        } else { $errorMsg = 'process_image.py not found'; }
                    } else {
                        // Preprocessing disabled: use raw image as both original and processed reference (no steps)
                        $rawData = file_get_contents($destPath);
                        if ($rawData !== false) {
                            $mime = mime_content_type($destPath);
                            if (!$mime) { $mime = 'image/jpeg'; }
                            $originalBase64 = base64_encode($rawData);
                            $processedBase64 = null; // explicitly null so UI can hide it
                            $steps = [];
                            // Pet type either user-selected or Unknown
                            if ($typeSelectedByUser) { $petType = $userPetType; $confidence = 0; }
                        } else {
                            $errorMsg = 'Failed to read uploaded image';
                        }
                    }

                    // Similarity: when preprocessing disabled, compare raw image file; else use processed
                    $queryImagePathForSimilarity = null;
                    if (!$errorMsg) {
                        if ($preprocessEnabled && $processedBase64) {
                            // Write processed to temp file later in similarity block
                            $queryImagePathForSimilarity = 'BASE64_PROCESSED';
                        } else {
                            // Use destPath (raw upload)
                            $queryImagePathForSimilarity = $destPath;
                        }
                    }
                    if ($queryImagePathForSimilarity && !$errorMsg) {
                        $computeScript = __DIR__ . DIRECTORY_SEPARATOR . 'python' . DIRECTORY_SEPARATOR . 'compute_matches.py';
                        $preDir = __DIR__ . DIRECTORY_SEPARATOR . 'Preprocessed';
                        if (file_exists($computeScript) && is_dir($preDir)) {
                            $tmpMatchesDir = sys_get_temp_dir();
                            $jpgPath = null;
                            if ($queryImagePathForSimilarity === 'BASE64_PROCESSED') {
                                $tmpQueryPath = tempnam($tmpMatchesDir, 'query_');
                                if ($tmpQueryPath) {
                                    $jpgPath = $tmpQueryPath . '.jpg';
                                    @rename($tmpQueryPath, $jpgPath);
                                    $raw = base64_decode($processedBase64);
                                    if ($raw !== false) { file_put_contents($jpgPath, $raw); }
                                }
                            } else {
                                $jpgPath = $queryImagePathForSimilarity; // raw file already exists
                            }
                            if ($jpgPath && file_exists($jpgPath)) {
                                // Always fetch a larger pool of matches; user will paginate client-side.
                                $computedType = $petType;
                                // If pet type is Unknown, keep it as Unknown (do not default to cat,dog)
                                if (!$typeSelectedByUser && !$preprocessEnabled && !preg_match('/^(cat|dog|unknown)/i',$petType)) {
                                    $computedType = 'cat,dog';
                                }
                                // Count gallery images across relevant folders to set dynamic capacity
                                $targetDirs = [];
                                if (preg_match('/unknown/i', $computedType)) { 
                                    $targetDirs[] = $preDir . DIRECTORY_SEPARATOR . 'Unknown'; 
                                } else {
                                    if (preg_match('/cat/i', $computedType)) { $targetDirs[] = $preDir . DIRECTORY_SEPARATOR . 'Cats'; }
                                    if (preg_match('/dog/i', $computedType)) { $targetDirs[] = $preDir . DIRECTORY_SEPARATOR . 'Dogs'; }
                                    if (!$targetDirs) { $targetDirs[] = $preDir . DIRECTORY_SEPARATOR . 'Unknown'; }
                                }
                                $fetchCount = 0;
                                foreach ($targetDirs as $td) {
                                    if (is_dir($td)) {
                                        foreach (scandir($td) as $fn) {
                                            if (preg_match('/\.(jpe?g|png|bmp|webp)$/i', $fn)) { $fetchCount++; }
                                        }
                                    }
                                }
                                // Fetch exactly the dataset size (or at least requestedTop if bigger)
                                $maxFetch = max($requestedTop, $fetchCount);
                                $argImage = escapeshellarg($jpgPath);
                                $argType = escapeshellarg($computedType);
                                $argDir = escapeshellarg($preDir);
                                $argTop = escapeshellarg((string)$maxFetch);
                                $cmd2 = $python . ' ' . escapeshellarg($computeScript) . ' ' . $argImage . ' ' . $argType . ' ' . $argDir . ' ' . $argTop . ' --debug';
                            } else {
                                $cmd2 = null;
                            }
                            if ($cmd2) {
                                // Execute CapsNet model
                                $output2 = shell_exec($cmd2 . ' 2>&1');
                                if ($output2) {
                                    $firstBrace2 = strpos($output2, '{');
                                    if ($firstBrace2 !== false) {
                                        $candidate2 = substr($output2, $firstBrace2);
                                        $json2 = json_decode($candidate2, true);
                                    } else {
                                        $json2 = json_decode($output2, true);
                                    }
                                    if ($json2 && isset($json2['ok']) && $json2['ok']) {
                                        $matches = $json2['matches'] ?? [];
                                        $similarityDebug = $json2['debug'] ?? [];
                                        $similarityAttempts = $json2['attempts'] ?? [];
                                        // Successful similarity means we clear earlier error (if any)
                                    } else {
                                        // Only set an error if we don't already have good matches
                                        if (!$matches) {
                                            $errorMsg = ($json2['error'] ?? 'Similarity computation failed');
                                        }
                                        $similarityDebug = $json2['debug'] ?? [];
                                        $similarityAttempts = $json2['attempts'] ?? [];
                                        if (!$json2) {
                                            $similarityDebug[] = 'RAW_OUTPUT: ' . substr($output2,0,8000);
                                        }
                                        $similarityDebug[] = 'COMMAND: ' . $cmd2;
                                    }
                                }
                                
                                // Execute Baseline model
                                $computeScriptBaseline = __DIR__ . DIRECTORY_SEPARATOR . 'python' . DIRECTORY_SEPARATOR . 'compute_matches_baseline.py';
                                if (file_exists($computeScriptBaseline)) {
                                    $cmdBaseline = $python . ' ' . escapeshellarg($computeScriptBaseline) . ' ' . $argImage . ' ' . $argType . ' ' . $argDir . ' ' . $argTop . ' --debug';
                                    $outputBaseline = shell_exec($cmdBaseline . ' 2>&1');
                                    if ($outputBaseline) {
                                        $firstBraceBaseline = strpos($outputBaseline, '{');
                                        if ($firstBraceBaseline !== false) {
                                            $candidateBaseline = substr($outputBaseline, $firstBraceBaseline);
                                            $jsonBaseline = json_decode($candidateBaseline, true);
                                        } else {
                                            $jsonBaseline = json_decode($outputBaseline, true);
                                        }
                                        if ($jsonBaseline && isset($jsonBaseline['ok']) && $jsonBaseline['ok']) {
                                            $matchesBaseline = $jsonBaseline['matches'] ?? [];
                                        } else {
                                            $matchesBaseline = [];
                                            if (!$jsonBaseline) {
                                                $similarityDebug[] = 'BASELINE_RAW: ' . substr($outputBaseline,0,2000);
                                            }
                                            $similarityDebug[] = 'BASELINE_CMD: ' . $cmdBaseline;
                                        }
                                    } else {
                                        $matchesBaseline = [];
                                    }
                                } else {
                                    $matchesBaseline = [];
                                    $similarityDebug[] = 'BASELINE_SCRIPT_NOT_FOUND: ' . $computeScriptBaseline;
                                }
                                
                                // Attempt to remove temp file only if it was the processed temp
                                if ($queryImagePathForSimilarity === 'BASE64_PROCESSED') { @unlink($jpgPath); }
                            } else {
                                $errorMsg = $errorMsg ?: 'Failed to create temporary file for similarity computation';
                            }
                        }
                    }
                    // After processing & similarity (success or fail) capture end time
                    $serverEndMs = microtime(true)*1000.0;
                    $elapsedTotalSec = 0;
                    if ($clientStartMs > 0 && $serverEndMs >= $clientStartMs) {
                        $elapsedTotalSec = ($serverEndMs - $clientStartMs)/1000.0; // full round-trip including upload
                    } else {
                        $elapsedTotalSec = ($serverEndMs - $serverStartMs)/1000.0; // fallback: server-only time
                    }
                    // Process time frontend output
                    $procLabel = 'Process Time: ' . number_format($elapsedTotalSec, 2) . ' secs';
                    $pillHtml = '<div style="background:#eef4ff;border:1px solid #c6d6f3;color:#1f3d7a;padding:6px 14px;border-radius:20px;font-size:0.72rem;font-weight:600;letter-spacing:.5px;display:inline-flex;align-items:center;gap:6px;white-space:nowrap;">'
                        . '<span style="display:inline-block;width:8px;height:8px;background:#4a8cff;border-radius:50%;box-shadow:0 0 0 3px rgba(74,140,255,0.15);"></span>'
                        . htmlspecialchars($procLabel) . '</div>';
                    echo '<script>document.addEventListener("DOMContentLoaded",function(){var p=document.getElementById("procTimePill"); if(p){p.innerHTML=' . json_encode($pillHtml) . ';}});</script>';
                }
                ?>

                <div class="search-image-frame">
                    <?php
                    if ($originalBase64) {
                        echo '<img src="data:image/jpeg;base64,' . htmlspecialchars($originalBase64) . '" alt="Search Image" class="search-image" />';
                    } elseif (isset($_FILES['pet-image']) && $_FILES['pet-image']['error'] === UPLOAD_ERR_OK) {
                        $imgData = file_get_contents($_FILES['pet-image']['tmp_name']);
                        $imgType = mime_content_type($_FILES['pet-image']['tmp_name']);
                        $base64 = 'data:' . $imgType . ';base64,' . base64_encode($imgData);
                        echo '<img src="' . $base64 . '" alt="Search Image" class="search-image" />';
                    } else { echo '<div class="search-image-placeholder">No Image Uploaded</div>'; }
                    ?>
                </div>
                <div class="preprocess-debug-card">
                    <div class="preprocess-debug-title">Preprocessed Image</div>
                    <div class="preprocess-debug-content">
                        <div class="preprocess-debug-row">
                            <div class="preprocess-debug-frame">
                                <?php
                                if (!$preprocessEnabled) {
                                    echo '<div style="font-size:0.75rem;color:#666;text-align:center;">(Preprocessing disabled)</div>';
                                } else {
                                    if ($processedBase64) {
                                        echo '<img src="data:image/jpeg;base64,' . htmlspecialchars($processedBase64) . '" alt="Detected & Resized 224x224">';
                                    } else {
                                        echo '<div style="font-size:0.75rem;color:#666;text-align:center;">No image</div>';
                                    }
                                }
                                ?>
                            </div>
                            <div class="preprocess-info">
                                <div class="preprocess-info-type">Pet Type: <?php echo htmlspecialchars($petType); ?><br>
                                    <?php if($typeSelectedByUser){
                                        echo 'Confidence: '.htmlspecialchars($petType).' (Selected by User)';
                                    } else {
                                        echo 'Confidence: '.htmlspecialchars(number_format($confidence,2)).'%';
                                    } ?>
                                </div>
                                <div class="preprocess-info-steps"><b>Preprocessing Steps:</b><br>
                                    <?php
                                    if ($steps) {
                                        foreach($steps as $idx=>$s){ echo ($idx+1) . '. ' . htmlspecialchars($s) . '<br>'; }
                                    } else { echo 'No steps recorded'; }
                                    if ($errorMsg) { 
                                        echo '<span style="color:#b30000;font-weight:700;word-wrap:break-word;word-break:break-word;max-width:100%;display:inline-block;">Error: ' . htmlspecialchars($errorMsg) . '</span>';
                                        if ($similarityDebug || $similarityAttempts) {
                                            echo '<details style="margin-top:6px;"><summary style="cursor:pointer;color:#223A7B;">Similarity Debug Log</summary><div style="font-size:0.7rem;max-height:160px;max-width:230px;overflow:auto;background:#FFFFFF;border:1px solid #B3C6FF;padding:6px;border-radius:6px;">';
                                            if ($similarityAttempts) {
                                                echo '<div><b>Load Attempts:</b> ' . htmlspecialchars(implode(', ', $similarityAttempts)) . '</div>';
                                            }
                                            if ($similarityDebug) {
                                                foreach ($similarityDebug as $d) { echo '<div>' . htmlspecialchars($d) . '</div>'; }
                                            }
                                            echo '</div></details>';
                                        }
                                    }
                                    ?>
                                </div>
                            </div>
                        </div>
                    </div>
                    <button style="background:#F6C23E;color:#FFFFFF;font-weight:700;border:none;border-radius:20px;padding:8px 24px;font-size:1rem;box-shadow:0 2px 8px rgba(60,90,200,0.10);cursor:pointer;align-self:flex-end;" onclick="window.location.href='find-missing-pet.php'">New search</button>
                </div>
            </div>
        </div>
        
        <!-- CapsNet Top Matches Section -->
        <div class="top-matches-section">
            <div class="top-matches-title" style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
                <span style="font-weight:700;color:#1b4fc1;">Proposed Model:</span>
                <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:200px;">
                    <input id="searchBoxCapsNet" type="text" placeholder="Search by filename..." 
                        style="flex:1;padding:6px 12px;border:1px solid #c2cee3;border-radius:8px;font-size:0.85rem;color:#223a7b;background:#ffffff;min-width:150px;"
                        oninput="handleSearchCapsNet()" />
                    <button id="clearSearchBtnCapsNet" type="button" 
                        style="background:#f0f4fb;border:1px solid #c2cee3;color:#223a7b;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:0.8rem;display:none;"
                        onclick="clearSearchCapsNet()">✕ Clear</button>
                </div>
                <div id="paginationControlsCapsNet" style="display:flex;align-items:center;gap:8px;">
                    <button id="prevPageBtnCapsNet" type="button" style="background:#f0f4fb;border:1px solid #c2cee3;color:#223a7b;border-radius:8px;padding:4px 10px;cursor:pointer;">◀</button>
                    <div style="display:flex;align-items:center;gap:4px;font-size:0.8rem;color:#223a7b;">
                        <span>Page</span>
                        <input id="pageInputCapsNet" type="number" value="1" min="1" style="width:45px;padding:2px 4px;border:1px solid #c2cee3;border-radius:6px;font-size:0.8rem;color:#223a7b;" />
                        <span id="totalPagesLabelCapsNet">/1</span>
                    </div>
                    <button id="nextPageBtnCapsNet" type="button" style="background:#f0f4fb;border:1px solid #c2cee3;color:#223a7b;border-radius:8px;padding:4px 10px;cursor:pointer;">▶</button>
                </div>
            </div>
            <div class="top-matches-row" id="topMatchesContainerCapsNet">
                <?php
                if (!($matches && count($matches) > 0)) {
                    echo '<div style="color:#4a5a7b;font-size:0.95rem;">No matches available from CapsNet model.</div>';
                }
                ?>
            </div>
        </div>
        
        <!-- Baseline CNN Top Matches Section -->
        <div class="top-matches-section" style="margin-top:32px;">
            <div class="top-matches-title" style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
                <span style="font-weight:700;color:#1b4fc1;">Baseline CNN:</span>
                <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:200px;">
                    <input id="searchBoxBaseline" type="text" placeholder="Search by filename..." 
                        style="flex:1;padding:6px 12px;border:1px solid #c2cee3;border-radius:8px;font-size:0.85rem;color:#223a7b;background:#ffffff;min-width:150px;"
                        oninput="handleSearchBaseline()" />
                    <button id="clearSearchBtnBaseline" type="button" 
                        style="background:#f0f4fb;border:1px solid #c2cee3;color:#223a7b;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:0.8rem;display:none;"
                        onclick="clearSearchBaseline()">✕ Clear</button>
                </div>
                <div id="paginationControlsBaseline" style="display:flex;align-items:center;gap:8px;">
                    <button id="prevPageBtnBaseline" type="button" style="background:#f0f4fb;border:1px solid #c2cee3;color:#223a7b;border-radius:8px;padding:4px 10px;cursor:pointer;">◀</button>
                    <div style="display:flex;align-items:center;gap:4px;font-size:0.8rem;color:#223a7b;">
                        <span>Page</span>
                        <input id="pageInputBaseline" type="number" value="1" min="1" style="width:45px;padding:2px 4px;border:1px solid #c2cee3;border-radius:6px;font-size:0.8rem;color:#223a7b;" />
                        <span id="totalPagesLabelBaseline">/1</span>
                    </div>
                    <button id="nextPageBtnBaseline" type="button" style="background:#f0f4fb;border:1px solid #c2cee3;color:#223a7b;border-radius:8px;padding:4px 10px;cursor:pointer;">▶</button>
                </div>
            </div>
            <div class="top-matches-row" id="topMatchesContainerBaseline">
                <?php
                if (!($matchesBaseline && count($matchesBaseline) > 0)) {
                    echo '<div style="color:#4a5a7b;font-size:0.95rem;">No matches available from Baseline CNN model.</div>';
                }
                ?>
            </div>
        </div>
        
        <!-- Shared Image Modal -->
        <div id="imageModal" style="position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:rgba(10,20,40,0.55);backdrop-filter:blur(3px);z-index:1000;">
            <div style="position:relative;background:#fff;padding:24px;border-radius:24px;width:90vmin;height:90vmin;max-width:800px;max-height:800px;min-width:280px;min-height:280px;box-shadow:0 10px 40px rgba(0,0,0,0.28);display:flex;flex-direction:column;">
                <button id="closeModalBtn" style="position:absolute;top:12px;right:12px;background:#223a7b;color:#fff;border:none;border-radius:50%;width:36px;height:36px;font-size:1.2rem;cursor:pointer;line-height:36px;text-align:center;z-index:2;">×</button>
                <div style="flex:1;background:#f8faff;display:flex;align-items:center;justify-content:center;border-radius:16px;margin-bottom:16px;overflow:hidden;">
                    <img id="modalImage" src="" alt="Enlarged Match" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:16px;transition:transform 0.3s ease;" onload="this.style.opacity='1'" onerror="this.style.opacity='1'" />
                </div>
                <div id="modalCaption" style="font-size:0.9rem;color:#223a7b;font-weight:600;text-align:center;word-break:break-word;padding:8px 16px;background:#f8faff;border-radius:12px;min-height:20px;"></div>
            </div>
        </div>
        <script>
        // ========== CapsNet Model JavaScript ==========
        const ALL_MATCHES_CAPSNET = <?php echo json_encode($matches); ?>;
        const PET_TYPE = <?php echo json_encode($petType); ?>;
        let PAGE_SIZE_CAPSNET = <?php echo intval($requestedTop); ?> || 3;
        let currentPageCapsNet = 1;
        let filteredMatchesCapsNet = ALL_MATCHES_CAPSNET;
        let searchQueryCapsNet = '';

        function handleSearchCapsNet(){
            const searchBox = document.getElementById('searchBoxCapsNet');
            const clearBtn = document.getElementById('clearSearchBtnCapsNet');
            searchQueryCapsNet = searchBox.value.toLowerCase().trim();
            
            if(searchQueryCapsNet === ''){
                filteredMatchesCapsNet = ALL_MATCHES_CAPSNET;
                clearBtn.style.display = 'none';
            } else {
                filteredMatchesCapsNet = ALL_MATCHES_CAPSNET.filter(m => {
                    const filename = filenameFromPath(m.path).toLowerCase();
                    return filename.includes(searchQueryCapsNet);
                });
                clearBtn.style.display = 'inline-block';
            }
            
            currentPageCapsNet = 1;
            renderMatchesCapsNet();
        }
        
        function clearSearchCapsNet(){
            const searchBox = document.getElementById('searchBoxCapsNet');
            const clearBtn = document.getElementById('clearSearchBtnCapsNet');
            searchBox.value = '';
            searchQueryCapsNet = '';
            filteredMatchesCapsNet = ALL_MATCHES_CAPSNET;
            clearBtn.style.display = 'none';
            currentPageCapsNet = 1;
            renderMatchesCapsNet();
        }
        
        function updatePaginationUICapsNet(){
            const totalPages = Math.max(1, Math.ceil((filteredMatchesCapsNet?filteredMatchesCapsNet.length:0)/PAGE_SIZE_CAPSNET));
            if(currentPageCapsNet>totalPages) currentPageCapsNet = totalPages;
            if(currentPageCapsNet<1) currentPageCapsNet = 1;
            const inp = document.getElementById('pageInputCapsNet');
            const totalLbl = document.getElementById('totalPagesLabelCapsNet');
            if(inp){ inp.value = currentPageCapsNet; inp.max = totalPages; }
            if(totalLbl) totalLbl.textContent = `/${totalPages}`;
            const prev = document.getElementById('prevPageBtnCapsNet');
            const next = document.getElementById('nextPageBtnCapsNet');
            if(prev) prev.disabled = currentPageCapsNet<=1;
            if(next) next.disabled = currentPageCapsNet>=totalPages;
        }
        
        function renderMatchesCapsNet(){
            const cont = document.getElementById('topMatchesContainerCapsNet');
            cont.innerHTML='';
            if(!filteredMatchesCapsNet || filteredMatchesCapsNet.length===0){
                if(searchQueryCapsNet){
                    cont.innerHTML = '<div style="color:#4a5a7b;font-size:0.95rem;">No CapsNet matches found for "' + searchQueryCapsNet + '".</div>';
                } else {
                    cont.innerHTML = '<div style="color:#4a5a7b;font-size:0.95rem;">No CapsNet matches available.</div>';
                }
                updatePaginationUICapsNet();
                return;
            }
            const start = (currentPageCapsNet-1)*PAGE_SIZE_CAPSNET;
            const subset = filteredMatchesCapsNet.slice(start, start+PAGE_SIZE_CAPSNET);
            subset.forEach(m => {
                const score = m.score || 0;
                const imgSrc = m.thumb_base64 ? 'data:image/jpeg;base64,'+m.thumb_base64 : 'assets/HomeCards/find-missing-pet-card.png';
                const fileName = filenameFromPath(m.path);
                const niceName = humanizeName(fileName);
                const card = document.createElement('div');
                card.className = 'top-match-card';
                card.style.borderRadius = '18px';
                const progressWidth = Math.min(100, Math.max(0, score));
                const detailsId = 'details_capsnet_'+m.rank+'_'+Math.random().toString(36).slice(2,8);
                
                let displayName = niceName;
                if(searchQueryCapsNet){
                    const regex = new RegExp('(' + searchQueryCapsNet.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
                    displayName = niceName.replace(regex, '<mark style="background:#fff59d;padding:1px 2px;border-radius:2px;">$1</mark>');
                }
                
                card.innerHTML = `
                    <div class="top-match-rank" style="background:linear-gradient(135deg,#1b4fc1,#4E73DF);">#${m.rank}</div>
                    <div class="top-match-image-wrapper">
                        <img src="${imgSrc}" class="top-match-image" alt="Match Image">
                    </div>
                    <div class="match-name" title="${fileName}">${displayName}</div>
                    <div style="font-size:0.85rem;color:#4a5a7b;margin-bottom:6px;">Type: ${PET_TYPE}</div>
                    <div style="display:flex;flex-direction:column;gap:4px;width:100%;">
                        <div style="font-size:0.78rem;color:#223a7b;font-weight:600;display:flex;justify-content:space-between;">
                            <span>Match Score</span><span>${score.toFixed(2)}%</span></div>
                        <div style="position:relative;height:8px;border-radius:6px;background:#dfe7f5;overflow:hidden;">
                            <div style="position:absolute;top:0;left:0;height:100%;width:${progressWidth}%;background:linear-gradient(90deg,#3867d6,#5b8bff);transition:width .4s;"></div>
                        </div>
                    </div>
                    <button class="details-btn" data-target="${detailsId}" style="background:#4E73DF;color:#fff;font-weight:600;border:none;border-radius:18px;padding:6px 16px;font-size:0.85rem;cursor:pointer;margin-top:10px;">View Details</button>
                    <button class="see-image-btn" data-img="${imgSrc}" data-full="${m.path || ''}" data-name="${fileName}" style="background:#f8faff;color:#4E73DF;font-weight:600;border:2px solid #4E73DF;border-radius:18px;padding:6px 16px;font-size:0.85rem;cursor:pointer;margin-top:0px;">See Image</button>
                    <div id="${detailsId}" style="display:none;margin-top:10px;font-size:0.68rem;color:#223a7b;background:#f0f4fb;padding:6px 8px;border:1px solid #d3deee;border-radius:8px;word-break:break-all;">
                        <div><b>File:</b> ${fileName}</div>
                        <div><b>Full Path:</b> ${m.path || 'n/a'}</div>
                    </div>
                `;
                cont.appendChild(card);
            });
            attachEventHandlers(cont);
            updatePaginationUICapsNet();
        }
        
        // ========== Baseline Model JavaScript ==========
        const ALL_MATCHES_BASELINE = <?php echo json_encode($matchesBaseline); ?>;
        let PAGE_SIZE_BASELINE = <?php echo intval($requestedTop); ?> || 3;
        let currentPageBaseline = 1;
        let filteredMatchesBaseline = ALL_MATCHES_BASELINE;
        let searchQueryBaseline = '';

        function handleSearchBaseline(){
            const searchBox = document.getElementById('searchBoxBaseline');
            const clearBtn = document.getElementById('clearSearchBtnBaseline');
            searchQueryBaseline = searchBox.value.toLowerCase().trim();
            
            if(searchQueryBaseline === ''){
                filteredMatchesBaseline = ALL_MATCHES_BASELINE;
                clearBtn.style.display = 'none';
            } else {
                filteredMatchesBaseline = ALL_MATCHES_BASELINE.filter(m => {
                    const filename = filenameFromPath(m.path).toLowerCase();
                    return filename.includes(searchQueryBaseline);
                });
                clearBtn.style.display = 'inline-block';
            }
            
            currentPageBaseline = 1;
            renderMatchesBaseline();
        }
        
        function clearSearchBaseline(){
            const searchBox = document.getElementById('searchBoxBaseline');
            const clearBtn = document.getElementById('clearSearchBtnBaseline');
            searchBox.value = '';
            searchQueryBaseline = '';
            filteredMatchesBaseline = ALL_MATCHES_BASELINE;
            clearBtn.style.display = 'none';
            currentPageBaseline = 1;
            renderMatchesBaseline();
        }
        
        function updatePaginationUIBaseline(){
            const totalPages = Math.max(1, Math.ceil((filteredMatchesBaseline?filteredMatchesBaseline.length:0)/PAGE_SIZE_BASELINE));
            if(currentPageBaseline>totalPages) currentPageBaseline = totalPages;
            if(currentPageBaseline<1) currentPageBaseline = 1;
            const inp = document.getElementById('pageInputBaseline');
            const totalLbl = document.getElementById('totalPagesLabelBaseline');
            if(inp){ inp.value = currentPageBaseline; inp.max = totalPages; }
            if(totalLbl) totalLbl.textContent = `/${totalPages}`;
            const prev = document.getElementById('prevPageBtnBaseline');
            const next = document.getElementById('nextPageBtnBaseline');
            if(prev) prev.disabled = currentPageBaseline<=1;
            if(next) next.disabled = currentPageBaseline>=totalPages;
        }
        
        function renderMatchesBaseline(){
            const cont = document.getElementById('topMatchesContainerBaseline');
            cont.innerHTML='';
            if(!filteredMatchesBaseline || filteredMatchesBaseline.length===0){
                if(searchQueryBaseline){
                    cont.innerHTML = '<div style="color:#4a5a7b;font-size:0.95rem;">No Baseline matches found for "' + searchQueryBaseline + '".</div>';
                } else {
                    cont.innerHTML = '<div style="color:#4a5a7b;font-size:0.95rem;">No Baseline matches available.</div>';
                }
                updatePaginationUIBaseline();
                return;
            }
            const start = (currentPageBaseline-1)*PAGE_SIZE_BASELINE;
            const subset = filteredMatchesBaseline.slice(start, start+PAGE_SIZE_BASELINE);
            subset.forEach(m => {
                const score = m.score || 0;
                const imgSrc = m.thumb_base64 ? 'data:image/jpeg;base64,'+m.thumb_base64 : 'assets/HomeCards/find-missing-pet-card.png';
                const fileName = filenameFromPath(m.path);
                const niceName = humanizeName(fileName);
                const card = document.createElement('div');
                card.className = 'top-match-card';
                card.style.borderRadius = '18px';
                const progressWidth = Math.min(100, Math.max(0, score));
                const detailsId = 'details_baseline_'+m.rank+'_'+Math.random().toString(36).slice(2,8);
                
                let displayName = niceName;
                if(searchQueryBaseline){
                    const regex = new RegExp('(' + searchQueryBaseline.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
                    displayName = niceName.replace(regex, '<mark style="background:#fff59d;padding:1px 2px;border-radius:2px;">$1</mark>');
                }
                
                card.innerHTML = `
                    <div class="top-match-rank" style="background:linear-gradient(135deg,#1b4fc1,#4E73DF);">#${m.rank}</div>
                    <div class="top-match-image-wrapper">
                        <img src="${imgSrc}" class="top-match-image" alt="Match Image">
                    </div>
                    <div class="match-name" title="${fileName}">${displayName}</div>
                    <div style="font-size:0.85rem;color:#4a5a7b;margin-bottom:6px;">Type: ${PET_TYPE}</div>
                    <div style="display:flex;flex-direction:column;gap:4px;width:100%;">
                        <div style="font-size:0.78rem;color:#223a7b;font-weight:600;display:flex;justify-content:space-between;">
                            <span>Match Score</span><span>${score.toFixed(2)}%</span></div>
                        <div style="position:relative;height:8px;border-radius:6px;background:#dfe7f5;overflow:hidden;">
                            <div style="position:absolute;top:0;left:0;height:100%;width:${progressWidth}%;background:linear-gradient(90deg,#3867d6,#5b8bff);transition:width .4s;"></div>
                        </div>
                    </div>
                    <button class="details-btn" data-target="${detailsId}" style="background:#4E73DF;color:#fff;font-weight:600;border:none;border-radius:18px;padding:6px 16px;font-size:0.85rem;cursor:pointer;margin-top:10px;">View Details</button>
                    <button class="see-image-btn" data-img="${imgSrc}" data-full="${m.path || ''}" data-name="${fileName}" style="background:#f8faff;color:#4E73DF;font-weight:600;border:2px solid #4E73DF;border-radius:18px;padding:6px 16px;font-size:0.85rem;cursor:pointer;margin-top:0px;">See Image</button>
                    <div id="${detailsId}" style="display:none;margin-top:10px;font-size:0.68rem;color:#223a7b;background:#f0f4fb;padding:6px 8px;border:1px solid #d3deee;border-radius:8px;word-break:break-all;">
                        <div><b>File:</b> ${fileName}</div>
                        <div><b>Full Path:</b> ${m.path || 'n/a'}</div>
                    </div>
                `;
                cont.appendChild(card);
            });
            attachEventHandlers(cont);
            updatePaginationUIBaseline();
        }
        
        // ========== Shared Utility Functions ==========
        function filenameFromPath(p){
            if(!p) return 'unknown';
            const parts = p.split(/[/\\\\]/);
            return parts[parts.length-1];
        }
        
        function humanizeName(fn){
            const base = fn.replace(/\.[^.]+$/, '').replace(/[_-]+/g,' ');
            return base.charAt(0).toUpperCase()+base.slice(1);
        }
        
        function attachEventHandlers(cont){
            // Attach toggle handlers for details buttons
            cont.querySelectorAll('.details-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const targetId = btn.getAttribute('data-target');
                    const box = document.getElementById(targetId);
                    if(!box) return;
                    const visible = box.style.display !== 'none';
                    box.style.display = visible ? 'none':'block';
                    btn.textContent = visible ? 'View Details' : 'Hide Details';
                });
            });
            
            // Image modal handlers
            cont.querySelectorAll('.see-image-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const modal = document.getElementById('imageModal');
                    const imgEl = document.getElementById('modalImage');
                    const cap = document.getElementById('modalCaption');
                    if(!modal||!imgEl) return;
                    const thumb = btn.getAttribute('data-img');
                    const fullPath = btn.getAttribute('data-full');
                    cap.textContent = btn.getAttribute('data-name') || '';
                    
                    let useSrc = thumb;
                    if(fullPath){
                        const relIdx = fullPath.lastIndexOf('Preprocessed');
                        if(relIdx !== -1){
                            const rel = fullPath.substring(relIdx).replace(/\\/g,'/');
                            useSrc = rel + '?v=' + Date.now();
                        }
                    }
                    imgEl.src = useSrc;
                    modal.style.display='flex';
                    document.body.style.overflow='hidden';
                    imgEl.style.opacity = '0';
                    imgEl.onload = () => {
                        const ratio = imgEl.naturalWidth / imgEl.naturalHeight;
                        if (ratio > 1.5) {
                            imgEl.style.width = '100%';
                            imgEl.style.height = 'auto';
                        } else if (ratio < 0.67) {
                            imgEl.style.width = 'auto';
                            imgEl.style.height = '100%';
                        } else {
                            imgEl.style.width = '100%';
                            imgEl.style.height = '100%';
                        }
                        
                        if (imgEl.naturalWidth < 256 && imgEl.naturalHeight < 256) {
                            imgEl.style.imageRendering = 'crisp-edges';
                            imgEl.style.maxWidth = '80%';
                            imgEl.style.maxHeight = '80%';
                            if (useSrc !== thumb) { imgEl.src = thumb; }
                        } else {
                            imgEl.style.imageRendering = 'auto';
                        }
                        imgEl.style.opacity = '1';
                    };
                });
            });
        }
        
        // ========== CapsNet Pagination Setup ==========
        const prevBtnCapsNet = document.getElementById('prevPageBtnCapsNet');
        const nextBtnCapsNet = document.getElementById('nextPageBtnCapsNet');
        if(prevBtnCapsNet) prevBtnCapsNet.addEventListener('click', ()=>{ 
            if(currentPageCapsNet>1){ currentPageCapsNet--; renderMatchesCapsNet(); }
        });
        if(nextBtnCapsNet) nextBtnCapsNet.addEventListener('click', ()=>{ 
            const totalPages = Math.max(1, Math.ceil((filteredMatchesCapsNet?filteredMatchesCapsNet.length:0)/PAGE_SIZE_CAPSNET)); 
            if(currentPageCapsNet<totalPages){ currentPageCapsNet++; renderMatchesCapsNet(); }
        });
        
        const pageInputCapsNet = document.getElementById('pageInputCapsNet');
        if(pageInputCapsNet){
            pageInputCapsNet.addEventListener('change', ()=>{
                const totalPages = Math.max(1, Math.ceil((filteredMatchesCapsNet?filteredMatchesCapsNet.length:0)/PAGE_SIZE_CAPSNET));
                let val = parseInt(pageInputCapsNet.value,10);
                if(isNaN(val)) val = 1;
                if(val<1) val=1; if(val>totalPages) val=totalPages;
                currentPageCapsNet = val;
                renderMatchesCapsNet();
            });
            pageInputCapsNet.addEventListener('keydown', (e)=>{
                if(e.key==='Enter'){
                    e.preventDefault();
                    pageInputCapsNet.dispatchEvent(new Event('change'));
                }
            });
        }
        
        // ========== Baseline Pagination Setup ==========
        const prevBtnBaseline = document.getElementById('prevPageBtnBaseline');
        const nextBtnBaseline = document.getElementById('nextPageBtnBaseline');
        if(prevBtnBaseline) prevBtnBaseline.addEventListener('click', ()=>{ 
            if(currentPageBaseline>1){ currentPageBaseline--; renderMatchesBaseline(); }
        });
        if(nextBtnBaseline) nextBtnBaseline.addEventListener('click', ()=>{ 
            const totalPages = Math.max(1, Math.ceil((filteredMatchesBaseline?filteredMatchesBaseline.length:0)/PAGE_SIZE_BASELINE)); 
            if(currentPageBaseline<totalPages){ currentPageBaseline++; renderMatchesBaseline(); }
        });
        
        const pageInputBaseline = document.getElementById('pageInputBaseline');
        if(pageInputBaseline){
            pageInputBaseline.addEventListener('change', ()=>{
                const totalPages = Math.max(1, Math.ceil((filteredMatchesBaseline?filteredMatchesBaseline.length:0)/PAGE_SIZE_BASELINE));
                let val = parseInt(pageInputBaseline.value,10);
                if(isNaN(val)) val = 1;
                if(val<1) val=1; if(val>totalPages) val=totalPages;
                currentPageBaseline = val;
                renderMatchesBaseline();
            });
            pageInputBaseline.addEventListener('keydown', (e)=>{
                if(e.key==='Enter'){
                    e.preventDefault();
                    pageInputBaseline.dispatchEvent(new Event('change'));
                }
            });
        }
        
        // ========== Modal Close Logic ==========
        const imageModal = document.getElementById('imageModal');
        const closeModalBtn = document.getElementById('closeModalBtn');
        if(closeModalBtn){
            closeModalBtn.addEventListener('click', ()=>{ 
                imageModal.style.display='none'; 
                document.body.style.overflow='auto'; 
            });
        }
        if(imageModal){
            imageModal.addEventListener('click', (e)=>{
                if(e.target === imageModal){ 
                    imageModal.style.display='none'; 
                    document.body.style.overflow='auto'; 
                }
            });
        }
        
        // ========== Initial Render ==========
        renderMatchesCapsNet();
        renderMatchesBaseline();
        </script>
    </main>
</body>
</html>
