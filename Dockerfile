version: "3.7"

services:
  carla:
    image: gitlab-extern.ivi.fraunhofer.de:4567/mission-control/carla/carla-rackwitz/carla_rackwitz:0.0.4
    runtime: nvidia
    restart: unless-stopped
    command: ["sh", "-c", "/home/carla/CarlaUE4.sh -RenderOffScreen"]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
        #- SDL_VIDEODRIVER=offscreen
        #- DISPLAY=0
        #user: "1001"
    volumes:
      - ./carla/DefaultEngine.ini:/home/carla/CarlaUE4/Config/DefaultEngine.ini
    privileged: true

  hero_1:
    image: gitlab-extern.ivi.fraunhofer.de:4567/mission-control/carla/hero/hero:3.6.7
    restart: unless-stopped
    environment:
      - CARLA_HOST=${CARLA_HOST}
      - CARLA_PORT=${CARLA_PORT}
      - AD_SUB_HOST=roi_1
      - PATHS_FILE=/carla/paths.json
      - SPAWN_GPS=51.4406548,12.3751764
      - START_DIST_MIN=100.
      - LOG_LEVEL=DEBUG
    volumes:
      - ./paths.json:/carla/paths.json
    depends_on:
      carla:
        condition: service_started
        restart: true

  roi_1:
    image: gitlab-extern.ivi.fraunhofer.de:4567/busrackwitz/remote-operation-interface/roi:tmp
    #image: gitlab-extern.ivi.fraunhofer.de:4567/busrackwitz/remote-operation-interface/roi:0.11.2
    restart: unless-stopped
    volumes:
      - ./roi_config/schema.json:/app/data/schema.json:ro
      - ./roi_config/timetable:/app/data/timetable:rw
    command: "./roi | grep -v go-zmq | grep -v go-rabbit"
    environment:
      - ROI_VEHICLE_ID=${SYSTEM_ID}
      - ROI_VEHICLE_SUB_ZMQ=tcp://hero_1:5555
      - ROI_STRATEGICPLAN_SCHEMA=./app/data/schema.json
      - ROI_SUPERVISOR_ID=0
      - ROI_SUPERVISOR_RABBIT_USER=${RMQ_USER}
      - ROI_SUPERVISOR_RABBIT_PASSWORD=${RMQ_PASSWORD}
      - ROI_SUPERVISOR_RABBIT_BROKERURL=${RMQ_BROKER_URL}
      - ROI_SUPERVISOR_RABBIT_VHOST=${RMQ_VHOST}
      - ROI_HEARTBEAT_RX_MS=1900
      - ROI_OFFLINEMODE_ENABLED=false
      - ROI_OFFLINEMODE_TRIPREQUESTRETRYMS=5000
      - ROI_OFFLINEMODE_TIMETABLE_FP=./app/data/timetable/timetable.json
      - ROI_OFFLINEMODE_TIMETABLE_TIMEOUTMS=10000
      - ROI_LOG_LEVEL=debug

  zmqrmq_1:
    image: gitlab-extern.ivi.fraunhofer.de:4567/busrackwitz/zmq2rmq:v1.2.9
    restart: unless-stopped
    volumes:
    - ${ZMQRMQ_CONF}:/app/data/app.yml:ro
    command: "./zmq2rmq -cfgPath /app/data/app.yml"
    environment:
      - Z2X_SYSTEMID=${SYSTEM_ID}
      - Z2X_RABBIT_USER=${RMQ_USER}
      - Z2X_RABBIT_PASSWORD=${RMQ_PASSWORD}
      - Z2X_RABBIT_BROKERURL=${RMQ_BROKER_URL}
      - Z2X_RABBIT_VHOST=${RMQ_VHOST}

  camera_1:
    image: gitlab-extern.ivi.fraunhofer.de:4567/mission-control/carla/carla-camera/camera:0.0.12
    restart: unless-stopped
    environment:
    - CARLA_HOST=${CARLA_HOST}
    - CARLA_PORT=${CARLA_PORT}
    - RMQ_BROKER_URL=${RMQ_BROKER_URL}
    - RMQ_USER=${RMQ_USER}
    - RMQ_PASSWORD=${RMQ_PASSWORD}
    - RMQ_VHOST=${RMQ_VHOST}
    - RMQ_EXCHANGE=xchange.dispogo.indications.vehicle.ul
    - RMQ_ROUTINGKEY=shuttle.40.camera.front.images
    - CAMERA_ATTACH_TO=hero1
    - CAMERA_LOCATION_Z=3.
    - CAMERA_LOCATION_X=0.
      # station camera (RSU OE211)
      #    - CAMERA_LOCATION_X=1289
      #    - CAMERA_LOCATION_Y=-1037
      #    - CAMERA_LOCATION_Z=126
      #    - CAMERA_YAW=270
      #    - CAMERA_PITCH=-30
      #    - CAMERA_QUALITY=web_maximum
      #    - CAMERA_WIDTH=1920
      #    - CAMERA_HEIGHT=1080
    depends_on:
      hero_1:
        condition: service_started
        restart: true
