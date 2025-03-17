docker-compose up --build -d

docker exec -it iotdashboard-mongo-1 /bin/sh

mongosh

rs.initiate();




2025-03-13 08:59:56 BadValue: security.keyFile is required when authorization is enabled with replica sets
2025-03-13 08:59:56 try 'mongod --help' for more information


C:\Users\facci>docker exec -it mongo mongosh --eval "rs.status()"
MongoServerError: no replset config has been received



docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" mongo1
