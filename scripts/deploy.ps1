# PowerShell Deployment Script for Agentic Trading Application
# Usage: .\scripts\deploy.ps1 [dev|prod|test]

param(
    [string]$Environment = "dev"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 Agentic Trading App Deployment (PowerShell)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Yellow
Write-Host ""

# Check prerequisites
function Check-Prerequisites {
    Write-Host "[INFO] Checking prerequisites..." -ForegroundColor Yellow
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Host "[ERROR] Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
        exit 1
    }
    
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Host "[ERROR] Docker Compose is not installed." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "[SUCCESS] Prerequisites check passed" -ForegroundColor Green
}

# Setup environment
function Setup-Environment {
    Write-Host "[INFO] Setting up environment..." -ForegroundColor Yellow
    
    if (-not (Test-Path ".env")) {
        Write-Host "[INFO] Creating .env from .env.example..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        
        if ($Environment -eq "prod") {
            # Generate secure secret key
            $secretKey = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | ForEach-Object {[char]$_})
            $content = Get-Content ".env"
            $content = $content -replace "AUTH_TOKEN_SECRET=.*", "AUTH_TOKEN_SECRET=$secretKey"
            Set-Content ".env" $content
            Write-Host "[SUCCESS] Generated secure AUTH_TOKEN_SECRET" -ForegroundColor Green
        }
    } else {
        Write-Host "[INFO] .env file already exists" -ForegroundColor Yellow
    }
}

# Build images
function Build-Images {
    Write-Host "[INFO] Building Docker images..." -ForegroundColor Yellow
    
    if ($Environment -eq "dev") {
        docker-compose build app frontend
    } else {
        docker-compose build --target production app
    }
    
    Write-Host "[SUCCESS] Docker images built successfully" -ForegroundColor Green
}

# Start services
function Start-Services {
    Write-Host "[INFO] Starting services..." -ForegroundColor Yellow
    
    if ($Environment -eq "dev") {
        docker-compose up -d app db redis frontend
    } else {
        docker-compose up -d app db redis
    }
    
    Write-Host "[SUCCESS] Services started" -ForegroundColor Green
}

# Health checks
function Invoke-HealthCheck {
    Write-Host "[INFO] Running health checks..." -ForegroundColor Yellow
    
    Start-Sleep -Seconds 10
    
    # Check API
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "[SUCCESS] API is healthy" -ForegroundColor Green
        } else {
            throw "API returned status code $($response.StatusCode)"
        }
    } catch {
        Write-Host "[ERROR] API health check failed: $_" -ForegroundColor Red
        docker-compose logs app
        exit 1
    }
    
    # Check database
    try {
        docker-compose exec -T db pg_isready -U postgres | Out-Null
        Write-Host "[SUCCESS] Database is healthy" -ForegroundColor Green
    } catch {
        Write-Host "[WARNING] Database health check failed (may need more time)" -ForegroundColor Yellow
    }
    
    # Check Redis
    try {
        docker-compose exec -T redis redis-cli ping | Out-Null
        Write-Host "[SUCCESS] Redis is healthy" -ForegroundColor Green
    } catch {
        Write-Host "[WARNING] Redis health check failed" -ForegroundColor Yellow
    }
}

# Show status
function Show-Status {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "📊 Service Status" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    docker-compose ps
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "🌐 Access Points" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "API:         http://localhost:8000"
    Write-Host "Frontend:    http://localhost:3000"
    Write-Host "Grafana:     http://localhost:3001 (admin/admin123)"
    Write-Host "Prometheus:  http://localhost:9090"
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "📝 Useful Commands" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "View logs:     docker-compose logs -f app"
    Write-Host "Stop services: docker-compose down"
    Write-Host "Restart:       docker-compose restart"
    Write-Host "Rebuild:       docker-compose build --no-cache"
    Write-Host ""
}

# Main deployment flow
function Main {
    Check-Prerequisites
    Setup-Environment
    
    switch ($Environment) {
        "dev" {
            Write-Host "[INFO] Starting development environment..." -ForegroundColor Yellow
            Build-Images
            Start-Services
            Invoke-HealthCheck
            Show-Status
            Write-Host "`n✅ Development environment ready!" -ForegroundColor Green
        }
        "prod" {
            Write-Host "[INFO] Starting production environment..." -ForegroundColor Yellow
            Build-Images
            Start-Services
            Invoke-HealthCheck
            Show-Status
            Write-Host "`n✅ Production deployment complete!" -ForegroundColor Green
        }
        "test" {
            Write-Host "[INFO] Running tests..." -ForegroundColor Yellow
            docker-compose run --rm app python -m pytest tests/
        }
        default {
            Write-Host "[ERROR] Unknown environment: $Environment" -ForegroundColor Red
            Write-Host "Usage: .\deploy.ps1 [dev|prod|test]"
            exit 1
        }
    }
}

# Execute main function
Main
