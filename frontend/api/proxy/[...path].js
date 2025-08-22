// Vercel Serverless proxy to forward /api/* to Railway backend
// Set RAILWAY_API_URL in Vercel project env, e.g. https://your-api.up.railway.app

export default async function handler(req, res) {
  try {
    const segments = req.query.path || [];
    const base = process.env.RAILWAY_API_URL;
    if (!base) {
      res.status(500).json({ error: 'RAILWAY_API_URL not configured' });
      return;
    }

    const target = new URL(base.replace(/\/?$/, '/') + segments.join('/'));
    if (req.url.includes('?')) {
      const q = req.url.split('?')[1];
      if (q) target.search = q;
    }

    // Build headers
    const headers = new Headers();
    for (const [key, value] of Object.entries(req.headers)) {
      if (key.toLowerCase() === 'host') continue;
      headers.set(key, Array.isArray(value) ? value.join(', ') : value ?? '');
    }
    headers.set('x-forwarded-by', 'vercel-proxy');

    // Read body if present
    const hasBody = !['GET', 'HEAD'].includes(req.method);
    let bodyBuffer = undefined;
    if (hasBody) {
      const chunks = [];
      await new Promise((resolve, reject) => {
        req.on('data', (c) => chunks.push(c));
        req.on('end', resolve);
        req.on('error', reject);
      });
      bodyBuffer = Buffer.concat(chunks);
    }

    const resp = await fetch(target.toString(), {
      method: req.method,
      headers,
      body: hasBody ? bodyBuffer : undefined,
    });

    // Copy status and headers
    res.status(resp.status);
    resp.headers.forEach((value, key) => {
      if (key.toLowerCase() === 'transfer-encoding') return;
      res.setHeader(key, value);
    });

    const arrayBuffer = await resp.arrayBuffer();
    res.send(Buffer.from(arrayBuffer));
  } catch (err) {
    res.status(502).json({ error: 'Proxy error', details: String(err) });
  }
}



