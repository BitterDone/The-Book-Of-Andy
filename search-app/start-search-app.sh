# # Needed the first time
# pip install podman-compose
# # Ensure an Active Systemd User Session
# # For Podman to function correctly in rootless mode, your user must have an active systemd user session. You can check this by running:
# loginctl | grep $(whoami)
# # If no session is listed, you can enable lingering for your user, which allows systemd user services to run even after logout:
# sudo loginctl enable-linger $(whoami)

# podman network create search-app-net
podman-compose down
# podman-compose build --no-cache
podman-compose build
podman-compose up
# podman-compose build
# podman-compose up --build
