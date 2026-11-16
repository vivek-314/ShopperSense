// Dependency-free local preview, equivalent to: python -m http.server --directory docs.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const root = process.env.SHOPPERSENSE_PREVIEW_ROOT ? path.resolve(process.env.SHOPPERSENSE_PREVIEW_ROOT) : path.resolve(__dirname, '../docs');
const port = Number(process.env.PORT || 8766);
const types = {'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.json':'application/json; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.csv':'text/csv; charset=utf-8','.zip':'application/zip'};
http.createServer((request, response) => {
  let pathname;
  try { pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname); }
  catch { response.writeHead(400).end('Bad URL'); return; }
  if (pathname.endsWith('/')) pathname += 'index.html';
  const target = path.resolve(root, '.' + pathname);
  if (!target.startsWith(root + path.sep)) { response.writeHead(403).end('Forbidden'); return; }
  fs.readFile(target, (error, data) => {
    if (error) { response.writeHead(404).end('Not found'); return; }
    response.writeHead(200, {'Content-Type': types[path.extname(target)] || 'application/octet-stream', 'Cache-Control': 'no-store'});
    response.end(data);
  });
}).listen(port, '127.0.0.1', () => console.log(`ShopperSense preview: http://127.0.0.1:${port}/`));
