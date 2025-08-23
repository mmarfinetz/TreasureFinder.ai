// Vercel Serverless proxy to forward /api/* to Railway backend
// Set RAILWAY_API_URL in Vercel project env, e.g. https://your-api.up.railway.app

export default async function handler(req, res) {
  const startTime = Date.now();
  
  try {
    const segments = req.query.path || [];
    const base = process.env.RAILWAY_API_URL;
    
    if (!base) {
      console.error('[Vercel Proxy] RAILWAY_API_URL not configured');
      res.status(500).json({ 
        error: 'Backend not configured',
        message: 'RAILWAY_API_URL environment variable is not set in Vercel',
        timestamp: new Date().toISOString()
      });
      return;
    }

    const target = new URL(base.replace(/\/?$/, '/') + segments.join('/'));
    if (req.url.includes('?')) {
      const q = req.url.split('?')[1];
      if (q) target.search = q;
    }

    console.log(`[Vercel Proxy] ${req.method} ${target.toString()}`);

    // Build headers
    const headers = new Headers();
    for (const [key, value] of Object.entries(req.headers)) {
      if (key.toLowerCase() === 'host') continue;
      headers.set(key, Array.isArray(value) ? value.join(', ') : value ?? '');
    }
    headers.set('x-forwarded-by', 'vercel-proxy');
    headers.set('x-original-path', req.url);

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

    // Add timeout for Railway requests
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120000); // 2 minute timeout

    const resp = await fetch(target.toString(), {
      method: req.method,
      headers,
      body: hasBody ? bodyBuffer : undefined,
      signal: controller.signal
    });

    clearTimeout(timeout);
    
    const elapsed = Date.now() - startTime;
    console.log(`[Vercel Proxy] Response: ${resp.status} in ${elapsed}ms`);

    // Copy status and headers
    res.status(resp.status);
    resp.headers.forEach((value, key) => {
      if (key.toLowerCase() === 'transfer-encoding') return;
      res.setHeader(key, value);
    });
    res.setHeader('x-proxy-elapsed-ms', elapsed.toString());

    const arrayBuffer = await resp.arrayBuffer();
    res.send(Buffer.from(arrayBuffer));
  } catch (err) {
    const elapsed = Date.now() - startTime;
    console.error(`[Vercel Proxy] Error after ${elapsed}ms:`, err);
    
    // Provide detailed error response
    const errorResponse = {
      error: 'Proxy error',
      message: err.message || String(err),
      elapsed_ms: elapsed,
      timestamp: new Date().toISOString()
    };
    
    // Add specific error context
    if (err.name === 'AbortError') {
      errorResponse.error = 'Request timeout';
      errorResponse.message = 'Railway backend took too long to respond (>120s)';
      res.status(504).json(errorResponse);
    } else if (err.message?.includes('ECONNREFUSED')) {
      errorResponse.error = 'Backend unreachable';
      errorResponse.message = 'Railway backend is not responding. Check if the service is running.';
      res.status(503).json(errorResponse);
    } else if (err.message?.includes('ETIMEDOUT')) {
      errorResponse.error = 'Connection timeout';
      errorResponse.message = 'Could not connect to Railway backend';
      res.status(504).json(errorResponse);
    } else {
      res.status(502).json(errorResponse);
    }
  }
}



