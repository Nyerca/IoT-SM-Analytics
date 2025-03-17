const express = require("express");
const mongoose = require("mongoose");
const WebSocket = require("ws");

const app = express();
const PORT = 3000;

mongoose.connect("mongodb://mongo1:27017,mongo2:27017,mongo3:27017/iot_db?replicaSet=rs0", {
    useNewUrlParser: true,
    useUnifiedTopology: true
});

const SensorSchema = new mongoose.Schema({}, { strict: false });
const SensorData = mongoose.model("SensorData", SensorSchema, "sensor_data");

app.use(express.static("public"));

const server = app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
const wss = new WebSocket.Server({ server });

wss.on("connection", (ws) => {
    console.log("Client connected to WebSocket");

    SensorData.watch().on("change", (change) => {
        if (change.operationType === "insert") {
            ws.send(JSON.stringify(change.fullDocument));
        }
    });
});

