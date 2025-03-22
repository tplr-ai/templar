import { Configuration } from 'webpack';
import * as webpackMerge from 'webpack-merge';
const { merge } = webpackMerge;
import CopyWebpackPlugin from 'copy-webpack-plugin';
import * as path from 'path';
import ESLintPlugin from 'eslint-webpack-plugin';
import ForkTsCheckerWebpackPlugin from 'fork-ts-checker-webpack-plugin';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const baseConfig = async (env: any): Promise<Configuration> => ({
  mode: env.production ? 'production' : 'development',

  output: {
    filename: '[name].js',
    path: path.join(__dirname, '../../dist'),
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
        { from: './img/*', to: '.' },
        { from: './README.md', to: '.' },
      ],
    }),
    new ESLintPlugin({
      extensions: ['.ts', '.tsx'],
      failOnError: false,
    }),
    new ForkTsCheckerWebpackPlugin({
      async: false,
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
            compilerOptions: {
              module: 'esnext',
            },
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

export default async (env: any) => {
  const baseOptions = await baseConfig(env);
  return merge(baseOptions, {
    entry: {
      module: './src/module.ts',
    },
  });
};