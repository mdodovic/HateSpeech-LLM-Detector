$ErrorActionPreference = "SilentlyContinue"

$configs = @(
    @{ freeze = "none" },
    @{ freeze = "backbone" },
    @{ freeze = "embeddings" },
    @{ freeze = "embeddings+3" },
    @{ freeze = "embeddings+6" },
    @{ freeze = "3" },
    @{ freeze = "6" }
)

$lr = "5e-5"
$bs = 8
$ml = 512

$total = $configs.Count
$i = 0

foreach ($c in $configs) {
    $i++
    $tag = "freeze_$($c.freeze)"
    $xlsx = "results/bertic/hp_binary_${tag}.xlsx"
    $log  = "results/bertic/hp_binary_${tag}.txt"

    Write-Host "`n===== RUN $i/$total : $tag =====" -ForegroundColor Cyan

    python finetune_bertic_binary.py `
        --lr $lr `
        --batch_size $bs `
        --max_length $ml `
        --freeze $c.freeze `
        --output $xlsx `
        --output_dir "hp_search_runs/$tag" `
        > $log 2>&1

    Write-Host "  Exit code: $LASTEXITCODE  ->  $log"
}

Write-Host "`n===== ALL $total RUNS COMPLETE =====" -ForegroundColor Green
