/**
 * Copyright 2016 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

var webpack = require('webpack');

var PROD = process.argv.indexOf('-p') !== -1

module.exports = {
	'context': __dirname,
	entry: {
		'Main': 'src/FeatureTest',
	},
	output: {
		filename: './build/[name].js',
		chunkFilename: './build/[id].js',
		sourceMapFilename : '[file].map',
	},
	resolve: {
		root: __dirname,
		modulesDirectories : ['node_modules', 'src', 'third_party', 'node_modules/tone', 'style'],
	},
	plugins: PROD ? [
	    new webpack.optimize.UglifyJsPlugin({minimize: true})
	  ] : [],
	 module: {
		loaders: [
			{
				test: /\.js$/,
				exclude: /(node_modules|Tone\.js)/,
				loader: 'babel', // 'babel-loader' is also a valid name to reference
				query: {
					presets: ['es2015']
				}
			},
			{
				test: /\.css$/,
				loader: 'style!css!autoprefixer!sass'
			},
			{
				test: /\.json$/,
				loader: 'json-loader'
			},
			{
				test: /\.(png|gif|svg)$/,
				loader: 'url-loader',
			},
			{
				test   : /\.(ttf|eot|woff(2)?)(\?[a-z0-9]+)?$/,
				loader : 'file-loader?name=images/font/[hash].[ext]'
			}
		]
	},
	devtool: PROD ? '' : '#eval-source-map'
};