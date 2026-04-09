$ErrorActionPreference = "SilentlyContinue"

# ── Fixed (best known values) ────────────────────────────────
$bs      = 8
$ml      = 512
$dropout = 0.3
$wd      = 0.05
$ls      = 0.05

# ── What we sweep ────────────────────────────────────────────
$configs = @(
    @{ lr = "1e-5"; freeze = "embeddings+2" },
    @{ lr = "2e-5"; freeze = "embeddings+2" },   # baseline
    @{ lr = "3e-5"; freeze = "embeddings+2" },
    @{ lr = "2e-5"; freeze = "embeddings" },
    @{ lr = "2e-5"; freeze = "embeddings+3" },
    @{ lr = "2e-5"; freeze = "3" },
    @{ lr = "2e-5"; freeze = "none" },            # compare vs freezing
    @{ lr = "3e-5"; freeze = "none" }
)

$total = $configs.Count
$i = 0

foreach ($c in $configs) {
    $i++
    $tag  = "lr$($c.lr)_freeze_$($c.freeze)"
    $xlsx = "results/bertic/hp_${tag}.xlsx"
    $log  = "results/bertic/hp_${tag}.txt"

    Write-Host "`n===== RUN $i/$total : $tag =====" -ForegroundColor Cyan

    python finetune_bertic_binary.py `
        --lr              $c.lr `
        --batch_size      $bs `
        --max_length      $ml `
        --freeze          $c.freeze `
        --dropout         $dropout `
        --weight_decay    $wd `
        --label_smoothing $ls `
        --output          $xlsx `
        --output_dir      "hp_runs/$tag" `
        > $log 2>&1

    Write-Host "  Exit: $LASTEXITCODE  ->  $log"
}

Write-Host "`n===== ALL $total RUNS COMPLETE =====" -ForegroundColor Green