$ErrorActionPreference = "SilentlyContinue"

$configs = @(
    @{ lr = "1e-5";  bs = 16; ml = 512 },
    @{ lr = "2e-5";  bs = 16; ml = 512 },  # baseline
    @{ lr = "3e-5";  bs = 16; ml = 512 },
    @{ lr = "5e-5";  bs = 16; ml = 512 },
    @{ lr = "1e-5";  bs = 8;  ml = 512 },
    @{ lr = "2e-5";  bs = 8;  ml = 512 },
    @{ lr = "3e-5";  bs = 8;  ml = 512 },
    @{ lr = "5e-5";  bs = 8;  ml = 512 },
    @{ lr = "1e-5";  bs = 16; ml = 256 },
    @{ lr = "2e-5";  bs = 16; ml = 256 },
    @{ lr = "3e-5";  bs = 16; ml = 256 },
    @{ lr = "5e-5";  bs = 16; ml = 256 },
    @{ lr = "1e-5";  bs = 8;  ml = 256 },
    @{ lr = "2e-5";  bs = 8;  ml = 256 },
    @{ lr = "3e-5";  bs = 8;  ml = 256 },
    @{ lr = "5e-5";  bs = 8;  ml = 256 }
)

$total = $configs.Count
$i = 0

foreach ($c in $configs) {
    $i++
    $tag = "lr$($c.lr)_bs$($c.bs)_ml$($c.ml)"
    $xlsx = "results/hp_binary_notreshold_${tag}.xlsx"
    $log  = "results/hp_binary_notreshold_${tag}.txt"

    Write-Host "`n===== RUN $i/$total : $tag =====" -ForegroundColor Cyan

    python finetune_bertic_binary.py `
        --lr $c.lr `
        --batch_size $c.bs `
        --max_length $c.ml `
        --output $xlsx `
        --output_dir "hp_search_runs_notreshold/$tag" `
        > $log 2>&1

    Write-Host "  Exit code: $LASTEXITCODE  ->  $log"
}

Write-Host "`n===== ALL $total RUNS COMPLETE =====" -ForegroundColor Green
