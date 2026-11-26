// Rasterise docs/assets/banner.svg to banner.png (1280x400) with resvg.
//   npm i @resvg/resvg-js   (anywhere on NODE_PATH)   then   node scripts/render_banner.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(process.env.RESVG_DIR ? `${process.env.RESVG_DIR}/` : import.meta.url);
const { Resvg } = require('@resvg/resvg-js');
const svg = readFileSync(new URL('../docs/assets/banner.svg', import.meta.url), 'utf8');
const png = new Resvg(svg, { fitTo: { mode: 'width', value: 1280 }, font: { loadSystemFonts: true } }).render().asPng();
writeFileSync(new URL('../docs/assets/banner.png', import.meta.url), png);
console.log(`banner.png ${png.length} bytes`);
