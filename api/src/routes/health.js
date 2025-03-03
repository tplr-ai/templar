const express = require('express');
const router = express.Router();
const { InfluxDB } = require('@influxdata/influxdb-client');
const influxConfig = require('../config/influxdb');

// Create the InfluxDB client and Query API instance.
const influxDB = new InfluxDB({ url: influxConfig.url, token: influxConfig.token });
const queryApi = influxDB.getQueryApi(influxConfig.org);

/**
 * A simple health check route.
 * This runs a minimal query to see if InfluxDB responds.
 */
router.get('/health', async (req, res) => {
    // The Flux query here is basic; adjust as needed.
    const fluxQuery = `
    from(bucket: "${influxConfig.bucket}")
      |> range(start: -1m)
      |> limit(n:1)
  `;
    console.debug("Running health check query:", fluxQuery);
    let results = [];
    try {
        await new Promise((resolve, reject) => {
            queryApi.queryRows(fluxQuery, {
                next(row, tableMeta) {
                    results.push(tableMeta.toObject(row));
                },
                error(err) {
                    console.error("Health query error:", err);
                    reject(err);
                },
                complete() {
                    resolve();
                },
            });
        });
        res.json({ status: "ok", results });
    } catch (err) {
        res.status(500).json({ status: "error", error: err.toString() });
    }
});

module.exports = router; 