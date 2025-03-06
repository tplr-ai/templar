const { InfluxDB } = require('@influxdata/influxdb-client');
const config = require('../config/influxdb');

class InfluxService {
  constructor() {
    this.client = new InfluxDB({ url: config.url, token: config.token });
    this.queryApi = this.client.getQueryApi(config.org);
  }

  async getLosses({ window = null, version = config.version, timeRange = '24h' }) {
    let fluxQuery = `
      from(bucket: "${config.bucket}")
        |> range(start: -${timeRange})
        |> filter(fn: (r) => r["_measurement"] == "templar_metrics_v2")
        |> filter(fn: (r) => r["role"] == "miner")
        |> filter(fn: (r) => r["version"] == "${version}")
        |> filter(fn: (r) => r["_field"] == "loss")
    `;

    if (window) {
      fluxQuery += `|> filter(fn: (r) => r["window"] == "${window}")`;
    }

    fluxQuery += `
        |> pivot(rowKey:["_time"], columnKey: ["uid"], valueColumn: "_value")
        |> sort(columns: ["_time"])
    `;

    try {
      const data = [];
      for await (const { values, tableMeta } of this.queryApi.iterateRows(fluxQuery)) {
        const o = tableMeta.toObject(values);
        // Convert the row into a more friendly format
        const timepoint = {
          time: o._time,
          window: o.window,
          losses: {}
        };

        // Extract losses for each UID
        Object.keys(o).forEach(key => {
          if (key.match(/^\d+$/)) { // If the key is a number (UID)
            timepoint.losses[key] = o[key];
          }
        });

        data.push(timepoint);
      }
      return data;
    } catch (error) {
      console.error('Error querying InfluxDB:', error);
      throw error;
    }
  }

  async getAvailableVersions() {
    const fluxQuery = `
      from(bucket: "${config.bucket}")
        |> range(start: -24h)
        |> filter(fn: (r) => r["_measurement"] == "templar_metrics_v2")
        |> filter(fn: (r) => r["role"] == "miner")
        |> group(columns: ["version"])
        |> distinct(column: "version")
    `;

    try {
      const versions = new Set();
      for await (const { values, tableMeta } of this.queryApi.iterateRows(fluxQuery)) {
        const o = tableMeta.toObject(values);
        versions.add(o.version);
      }
      return Array.from(versions);
    } catch (error) {
      console.error('Error querying versions from InfluxDB:', error);
      throw error;
    }
  }

  async getTokensPerSec({ uid = null, version = config.version, timeRange = '24h', window = null, aggregate = false }) {
    let fluxQuery = `
      from(bucket: "${config.bucket}")
        |> range(start: -${timeRange})
        |> filter(fn: (r) => r["_measurement"] == "templar_metrics_v2")
        |> filter(fn: (r) => r["role"] == "miner")
        |> filter(fn: (r) => r["version"] == "${version}")
        |> filter(fn: (r) => r["_field"] == "tokens_per_sec")
    `;

    if (window) {
      fluxQuery += `|> filter(fn: (r) => r["window"] == "${window}")`;
    }

    if (uid) {
      fluxQuery += `|> filter(fn: (r) => r["uid"] == "${uid}")`;
    }

    if (aggregate) {
      fluxQuery += `
        |> group(columns: ["_time"])
        |> mean()
      `;
    }

    fluxQuery += `|> sort(columns: ["_time"])`;

    try {
      const data = [];
      for await (const { values, tableMeta } of this.queryApi.iterateRows(fluxQuery)) {
        const o = tableMeta.toObject(values);
        data.push({
          time: o._time,
          tokens_per_sec: o._value,
          ...(aggregate ? {} : { uid: o.uid })
        });
      }

      return {
        success: true,
        data: data,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      console.error('InfluxDB Query Error:', error);
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }
}

module.exports = new InfluxService();