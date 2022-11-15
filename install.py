import launch

if not launch.is_installed("scikit_learn"):
    launch.run_pip("install scikit_learn", "requirements for scikit_learn")