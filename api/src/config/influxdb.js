require('dotenv').config();

const influxConfig = {
  url: 'https://uaepr2itgl-tzxeth774u3fvf.timestream-influxdb.us-east-2.on.aws:8086',
  token: process.env.INFLUXDB_TOKEN,
  org: 'templar',
  bucket: 'tplr',
  version: '0.2.29t' // Match version from tplr/__init__.py
};

module.exports = influxConfig;