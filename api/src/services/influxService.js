import { InfluxDB } from '@influxdata/influxdb-client';
import influxConfig from '../config/influxdb.js';

class InfluxService {
  constructor() {
    this.client = new InfluxDB({ url: influxConfig.url, token: influxConfig.token });
    this.queryApi = this.client.getQueryApi(influxConfig.org);
  }

  /**
   * Query InfluxDB with timeout
   * @param {string} query - Flux query string
   * @param {number} timeoutMs - Timeout in milliseconds
   * @returns {Promise<Array>} - Query results
   */
  async queryWithTimeout(query, timeoutMs = 60000) {
    return new Promise((resolve, reject) => {
      const results = [];
      const startTime = Date.now();
      console.debug("=== Starting query ===");
      console.debug("Timeout set to:", timeoutMs, "ms");
      console.debug("Query:", query);

      const timer = setTimeout(() => {
        const elapsed = Date.now() - startTime;
        console.error(`Query timed out after ${elapsed} ms`);
        reject(new Error('Query timed out'));
      }, timeoutMs);

      this.queryApi.queryRows(query, {
        next(row, tableMeta) {
          const elapsed = Date.now() - startTime;
          const obj = tableMeta.toObject(row);
          results.push(obj);
          console.debug(`Row received at ${elapsed} ms`);
        },
        error(err) {
          clearTimeout(timer);
          const elapsed = Date.now() - startTime;
          console.error(`Error after ${elapsed} ms:`, err);
          reject(err);
        },
        complete() {
          clearTimeout(timer);
          const duration = Date.now() - startTime;
          console.debug(`Query complete after ${duration} ms; total rows: ${results.length}`);
          resolve(results);
        },
      });
    });
  }

  /**
   * Format response for consistent API output
   * @param {Array} results - Query results
   * @param {Object} keyMappings - Map of keys to rename
   * @returns {Object} Formatted response
   */
  formatResponse(results, keyMappings = {}) {
    return {
      success: true,
      data: results.map(row => {
        const formattedRow = {
          timestamp: row._time
        };
        
        // Apply any key mappings
        for (const [key, newKey] of Object.entries(keyMappings)) {
          if (row.hasOwnProperty(key)) {
            formattedRow[newKey] = row[key];
          }
        }
        
        // Add all remaining fields
        for (const [key, value] of Object.entries(row)) {
          // Skip metadata fields and already mapped keys
          if (!key.startsWith('_') && !keyMappings[key] && !formattedRow[key]) {
            formattedRow[key] = value;
          }
        }
        
        // Always include the value
        if (row.hasOwnProperty('_value') && !formattedRow.value) {
          formattedRow.value = row._value;
        }
        
        return formattedRow;
      }),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Query metrics by measurement name with filtering
   * @param {string} measurement - Measurement name
   * @param {Object} filters - Key-value pairs to filter on
   * @param {string} timeRange - Time range (e.g., '-1h')
   * @returns {Promise<Object>} Formatted query results
   */
  async queryMetrics(measurement, filters = {}, timeRange = '-2h') {
    let fluxQuery = `
      from(bucket: "${influxConfig.bucket}")
        |> range(start: ${timeRange})
        |> filter(fn: (r) => r["_measurement"] == "${measurement}")
    `;
    
    // Apply all filters
    for (const [key, value] of Object.entries(filters)) {
      if (value !== undefined && value !== null && value !== '') {
        // Handle special case for _field
        if (key === 'field') {
          fluxQuery += `|> filter(fn: (r) => r["_field"] == "${value}")`;
        } else {
          fluxQuery += `|> filter(fn: (r) => r["${key}"] == "${value}")`;
        }
      }
    }
    
    fluxQuery += `|> last()`;
    
    try {
      const results = await this.queryWithTimeout(fluxQuery);
      return this.formatResponse(results);
    } catch (error) {
      console.error(`Error querying ${measurement}:`, error);
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Legacy method to get losses
   */
  async getLosses({ window = null, version = influxConfig.version, timeRange = '24h' }) {
    let fluxQuery = `
      from(bucket: "${influxConfig.bucket}")
        |> range(start: -${timeRange})
        |> filter(fn: (r) => r["_measurement"] == "training_step" OR r["_measurement"] == "templar_metrics_v2")
        |> filter(fn: (r) => r["_field"] == "loss")
    `;

    if (window) {
      fluxQuery += `|> filter(fn: (r) => r["window"] == "${window}")`;
    }

    if (version) {
      fluxQuery += `|> filter(fn: (r) => r["version"] == "${version}")`;
    }

    fluxQuery += `
        |> pivot(rowKey:["_time"], columnKey: ["uid"], valueColumn: "_value")
        |> sort(columns: ["_time"])
    `;

    try {
      const data = [];
      const results = await this.queryWithTimeout(fluxQuery);
      
      for (const row of results) {
        // Convert the row into a more friendly format
        const timepoint = {
          time: row._time,
          window: row.window,
          losses: {}
        };

        // Extract losses for each UID
        Object.keys(row).forEach(key => {
          if (key.match(/^\d+$/)) { // If the key is a number (UID)
            timepoint.losses[key] = row[key];
          }
        });

        data.push(timepoint);
      }
      
      return {
        success: true,
        data,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error querying InfluxDB:', error);
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Get available versions from the metrics
   */
  async getAvailableVersions() {
    const fluxQuery = `
      from(bucket: "${influxConfig.bucket}")
        |> range(start: -24h)
        |> filter(fn: (r) => r["_measurement"] == "training_step" OR r["_measurement"] == "templar_metrics_v2")
        |> group(columns: ["version"])
        |> distinct(column: "version")
    `;

    try {
      const versions = new Set();
      const results = await this.queryWithTimeout(fluxQuery);
      
      for (const row of results) {
        if (row.version) {
          versions.add(row.version);
        }
      }
      
      return {
        success: true,
        data: Array.from(versions),
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error querying versions from InfluxDB:', error);
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Legacy method to get tokens per second
   */
  async getTokensPerSec({ uid = null, version = influxConfig.version, timeRange = '24h', window = null, aggregate = false }) {
    let fluxQuery = `
      from(bucket: "${influxConfig.bucket}")
        |> range(start: -${timeRange})
        |> filter(fn: (r) => r["_measurement"] == "training_step" OR r["_measurement"] == "templar_metrics_v2")
        |> filter(fn: (r) => r["_field"] == "tokens_per_sec")
    `;

    if (version) {
      fluxQuery += `|> filter(fn: (r) => r["version"] == "${version}")`;
    }

    if (window) {
      fluxQuery += `|> filter(fn: (r) => r["window"] == "${window}")`;
    }

    if (uid) {
      fluxQuery += `|> filter(fn: (r) => r["uid"] == "${uid}")`;
    }

    if (aggregate) {
      fluxQuery += `
        |> group()
        |> mean()
      `;
    }

    fluxQuery += `|> sort(columns: ["_time"])`;

    try {
      const results = await this.queryWithTimeout(fluxQuery);
      return this.formatResponse(results, { '_value': 'tokens_per_sec' });
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

// Export a singleton instance
export default new InfluxService();