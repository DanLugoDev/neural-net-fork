const path = require('path')
const CleanWebpackPlugin = require('clean-webpack-plugin')
const webpack = require('webpack')
const nodeExternals = require('webpack-node-externals')

const { name, version } = require('./package.json')

module.exports = {
  entry: './src/Main.ts',
  devtool: 'inline-source-map',
  target: 'node',
  externals: [nodeExternals()],
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/
      }
    ]
  },
  plugins: [
    new CleanWebpackPlugin(['dist'])
  ],
  resolve: {
    extensions: [".ts"]
  },
  output: {
    filename: `${name}-${version}.js`,
    path: path.resolve(__dirname, 'dist')
  }
}
