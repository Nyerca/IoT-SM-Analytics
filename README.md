# Iot-SM-Analytics IoT Smart Maintenance Analytics

## Description
- Sensors sends data to a kafka topic
- Consumers get the data from kafka topic and saves them to MongoDb
- Node.js app reacts with change stream to MongoDbs insertions
- Predictive Maintenance done on the data

### How to run the project
docker-compose up --build -d


### Notes (commands and problems)
docker exec -it iotdashboard-mongo-1 /bin/sh

mongosh

rs.initiate();


2025-03-13 08:59:56 BadValue: security.keyFile is required when authorization is enabled with replica sets
2025-03-13 08:59:56 try 'mongod --help' for more information

C:\Users\facci>docker exec -it mongo mongosh --eval "rs.status()"
MongoServerError: no replset config has been received


docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" mongo1
