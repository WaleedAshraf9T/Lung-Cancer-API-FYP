# run_server.py
import uvicorn
import argparse
import os

def run_server(host="0.0.0.0", port=8000, reload=False, workers=1, env="development"):
    """
    Run the FastAPI server with specified configuration
    """
    # Set environment variables
    os.environ["ENV"] = env
    
    if not os.environ.get("API_KEY"):
        print("⚠️  Warning: API_KEY environment variable not set")
        print("Setting a default API key for development...")
        os.environ["API_KEY"] = "development-key"
    
    # Print server information
    print(f"\n🚀 Starting Lung Cancer Detection API Server")
    print(f"├── Environment: {env}")
    print(f"├── Host: {host}")
    print(f"├── Port: {port}")
    print(f"├── Workers: {workers}")
    print(f"└── Auto-reload: {reload}\n")
    
    # Configure uvicorn
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Lung Cancer Detection API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--env", default="development", choices=["development", "production"], 
                       help="Environment (development/production)")
    
    args = parser.parse_args()
    run_server(args.host, args.port, args.reload, args.workers, args.env)