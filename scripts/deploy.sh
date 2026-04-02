#!/bin/bash
# Production Deployment Script for Agentic Trading Application
# Usage: ./deploy.sh [dev|prod|test]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENV=${1:-dev}
PROJECT_NAME="agentic-trading-app"

echo "========================================"
echo "🚀 Agentic Trading App Deployment"
echo "========================================"
echo "Environment: $ENV"
echo ""

# Function to print status
print_status() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    if [ ! -f .env ]; then
        print_status "Creating .env from .env.example..."
        cp .env.example .env
        
        # Generate secure secret key for production
        if [ "$ENV" = "prod" ]; then
            SECRET_KEY=$(openssl rand -hex 32)
            sed -i "s/AUTH_TOKEN_SECRET=.*/AUTH_TOKEN_SECRET=$SECRET_KEY/" .env
            print_success "Generated secure AUTH_TOKEN_SECRET"
        fi
    else
        print_status ".env file already exists"
    fi
}

# Build images
build_images() {
    print_status "Building Docker images..."
    
    if [ "$ENV" = "dev" ]; then
        docker-compose build app frontend
    else
        docker-compose build --target production app
    fi
    
    print_success "Docker images built successfully"
}

# Start services
start_services() {
    print_status "Starting services..."
    
    if [ "$ENV" = "dev" ]; then
        # Development: start all services
        docker-compose up -d app db redis frontend
    else
        # Production: start only essential services
        docker-compose up -d app db redis
    fi
    
    print_success "Services started"
}

# Run health checks
health_check() {
    print_status "Running health checks..."
    
    # Wait for services to be ready
    sleep 10
    
    # Check API health
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        print_success "API is healthy"
    else
        print_error "API health check failed"
        docker-compose logs app
        exit 1
    fi
    
    # Check database health
    if docker-compose exec -T db pg_isready -U postgres > /dev/null 2>&1; then
        print_success "Database is healthy"
    else
        print_warning "Database health check failed (may need more time)"
    fi
    
    # Check Redis health
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        print_success "Redis is healthy"
    else
        print_warning "Redis health check failed"
    fi
}

# Show status
show_status() {
    echo ""
    echo "========================================"
    echo "📊 Service Status"
    echo "========================================"
    docker-compose ps
    
    echo ""
    echo "========================================"
    echo "🌐 Access Points"
    echo "========================================"
    echo "API:       http://localhost:8000"
    echo "Frontend:  http://localhost:3000"
    echo "Grafana:   http://localhost:3001 (admin/admin123)"
    echo "Prometheus: http://localhost:9090"
    echo ""
    echo "========================================"
    echo "📝 Useful Commands"
    echo "========================================"
    echo "View logs:     docker-compose logs -f app"
    echo "Stop services: docker-compose down"
    echo "Restart:       docker-compose restart"
    echo "Rebuild:       docker-compose build --no-cache"
    echo ""
}

# Main deployment flow
main() {
    check_prerequisites
    setup_environment
    
    case "$ENV" in
        dev)
            print_status "Starting development environment..."
            build_images
            start_services
            health_check
            show_status
            print_success "✅ Development environment ready!"
            ;;
        prod)
            print_status "Starting production environment..."
            build_images
            start_services
            health_check
            show_status
            print_success "✅ Production deployment complete!"
            ;;
        test)
            print_status "Running tests..."
            docker-compose run --rm app python -m pytest tests/
            ;;
        *)
            print_error "Unknown environment: $ENV"
            echo "Usage: ./deploy.sh [dev|prod|test]"
            exit 1
            ;;
    esac
}

# Run main function
main
