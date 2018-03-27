# MusicVAE Deeplearn.js API

This JavaScript implementation of [MusicVAE](https://g.co/magenta/music-vae) uses [Deeplearn.js](https://deeplearnjs.org) for GPU-accelerated inference.

For the Python TensorFlow implementation, see the [main Magenta repo](https://goo.gl/magenta/musicvae-code).

## Usage

To use in your application, install the npm package [@magenta/music-vae](https://www.npmjs.com/package/@magenta/music-vae), or use the [pre-built bundle](https://unpkg.com/@magenta/music-vae).

You can then instantiate a `MusicVAE` object with:

```js
let mvae = new MusicVAE('/path/to/checkpoint')
```

For a complete guide on how to build an app with MusicVAE, read the [Melody Mixer tutorial][cl-tutorial].

## Pre-trained Checkpoints

Several pre-trained MusicVAE checkpoints are hosted on GCS. While we do not plan to remove any of the current checkpoints, we will be adding more in the future, so your applications should reference the [checkpoints.json](https://goo.gl/magenta/musicvae-checkpoints) file to see which checkpoints are available.

If your application has a high QPS, you must mirror these files on your own server.

## Example Applications

* [Beat Blender](https://g.co/beatblender) by [Google Creative Lab](https://github.com/googlecreativelab)
* [Melody Mixer](https://g.co/melodymixer) by [Google Creative Lab](https://github.com/googlecreativelab)
* [Latent Loops](https://goo.gl/magenta/latent-loops) by [Google Pie Shop](https://github.com/teampieshop)
* [Neural Drum Machine](https://codepen.io/teropa/pen/RMGxOQ) by [Tero Parviainen](https://github.com/teropa)

## Example Commands

`yarn install` to install dependencies.

`yarn build` to produce a commonjs version with typescript definitions for MusicVAE in the `es5/` folder that can then be consumed by others over NPM.

`yarn bundle` to produce a bundled version in `dist/`.

`yarn run-demo` to build and run the demo.

[cl-tutorial]: https://medium.com/@torinblankensmith/8ad5b42b4d0b
