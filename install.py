import launch

if not launch.is_installed("runpod"):
    launch.run_pip("install runpod", "requirements for RUNPOD Serverless")