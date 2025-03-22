const path = require('path');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const { merge } = require('webpack-merge');

const baseConfig = (env) => ({
  mode: env.production ? 'production' : 'development',

  output: {
    filename: '[name].js',
    path: path.join(__dirname, 'dist'),
    libraryTarget: 'amd',
    clean: true,
  },

  externals: [
    '@emotion/react',
    '@emotion/css',
    'react',
    'react-dom',
    '@grafana/data',
    '@grafana/runtime',
    '@grafana/ui',
  ],

  resolve: {
    extensions: ['.ts', '.tsx', '.js', '.jsx'],
  },

  plugins: [
    new CopyWebpackPlugin({
      patterns: [
        { from: './src/plugin.json', to: '.' },
        { from: './img', to: './img' },
        { from: './README.md', to: '.' },
      ],
    }),
  ],

  module: {
    rules: [
      {
        test: /\.[tj]sx?$/,
        use: {
          loader: 'ts-loader',
          options: {
            transpileOnly: true,
          },
        },
        exclude: /node_modules/,
      },
      {
        test: /\.s?css$/,
        use: ['style-loader', 'css-loader', 'sass-loader'],
      },
      {
        test: /\.(png|jpe?g|gif|svg)$/,
        type: 'asset/resource',
      },
    ],
  },
});

module.exports = (env) => {
  const options = baseConfig(env);
  return merge(options, {
    entry: {
      module: './src/module.ts',
    },
  });
};