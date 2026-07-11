import { RequestInfo, RequestInit } from 'typescript/lib/lib.dom';

/**
 * Configuration for the Barewire fetch interceptor.
 */
interface BarewireConfig {
  /**
   * The base URL of your Barewire edge proxy (e.g., 'https://my-barewire-worker.workers.dev').
   * This is a mandatory field.
   */
  proxyUrl: string;
  /**
   * Optional: API key for authenticating with your Barewire proxy.
   * If provided, it will be sent as an `Authorization: Bearer <apiKey>` header.
   */
  apiKey?: string;
  /**
   * Optional: Enable console logging for debugging purposes.
   */
  debug?: boolean;
  /**
   * Optional: A list of URL patterns (strings or RegExp) to exclude from proxying.
   * Requests matching these patterns will be sent directly via the original fetch.
   * String patterns will use `String.prototype.includes()`.
   */
  excludeUrls?: (string | RegExp)[];
}

// Global variables to store the original fetch reference and the current configuration.
// This allows restoring fetch and checking config (e.g., for excludeUrls) within the interceptor.
let originalFetch: typeof fetch | null = null;
let currentBarewireConfig: BarewireConfig | null = null;

/**
 * Sets up a fetch interceptor to route specified requests through a Barewire edge proxy.
 * This is useful for adding authentication, logging, or other edge logic via Barewire.
 *
 * @param config Configuration for the Barewire proxy.
 * @returns A function to call to restore the original fetch behavior.
 */
export function setupBarewireInterceptor(config: BarewireConfig): () => void {
  if (!config.proxyUrl) {
    console.error('BarewireInterceptor: config.proxyUrl is required.');
    return () => {}; // Return a no-op cleanup function
  }

  // If fetch is already intercepted, restore it before setting up a new interception
  if (originalFetch !== null) {
    console.warn('BarewireInterceptor: fetch is already intercepted. Restoring original fetch before re-intercepting.');
    restoreBarewireInterceptor();
  }

  originalFetch = window.fetch;
  currentBarewireConfig = config; // Store config for use by the interceptor and restore function

  /**
   * Overrides the global `fetch` function.
   * All subsequent fetch calls will be handled by this custom function.
   */
  window.fetch = async (input: RequestInfo, init?: RequestInit): Promise<Response> => {
    let originalRequest: Request;

    // Create a Request object that fully represents the intended request.
    // The Request constructor correctly merges properties: if `input` is a Request,
    // `init` properties override `input` properties. If `input` is a string, `init` is used.
    // This also implicitly handles cloning the request's body if `input` was a Request object.
    try {
      originalRequest = new Request(input, init);
    } catch (e) {
      console.error(`[BarewireInterceptor] Failed to construct original Request object for ${input}:`, e);
      // Fallback: If construction fails (e.g., invalid URL), use original fetch.
      // This is a safety measure; typically `new Request` should work for valid inputs.
      return originalFetch!(input, init);
    }

    const requestUrl = originalRequest.url;

    // Check if the URL should be excluded from proxying based on configuration.
    if (currentBarewireConfig?.excludeUrls) {
      const shouldExclude = currentBarewireConfig.excludeUrls.some(pattern => {
        if (typeof pattern === 'string') {
          return requestUrl.includes(pattern);
        } else { // RegExp
          return pattern.test(requestUrl);
        }
      });
      if (shouldExclude) {
        if (config.debug) {
          console.log(`[BarewireInterceptor] Excluding URL from proxy: ${requestUrl}`);
        }
        return originalFetch!(input, init); // Use the original fetch for excluded URLs
      }
    }

    if (config.debug) {
      console.log(`[BarewireInterceptor] Intercepting request to: ${requestUrl}`);
    }

    // Prepare headers for the request that will be sent to the Barewire proxy.
    const proxyHeaders = new Headers(originalRequest.headers);

    // Add Barewire-specific headers required by the proxy.
    // The `X-Barewire-Target-URL` header tells the Barewire worker where to forward the request.
    proxyHeaders.set('X-Barewire-Target-URL', requestUrl);

    // If an API key is provided, add it for authentication with the Barewire proxy.
    if (config.apiKey) {
      proxyHeaders.set('Authorization', `Bearer ${config.apiKey}`);
    }

    // Construct the RequestInit object for the actual call to the Barewire proxy.
    // We use properties from the `originalRequest` to ensure all aspects of the original
    // request (method, body, credentials, etc.) are preserved.
    const proxyRequestInit: RequestInit = {
      method: originalRequest.method,
      headers: proxyHeaders,
      body: originalRequest.body, // The body from `originalRequest` is correctly prepared (e.g., cloned)
      credentials: originalRequest.credentials,
      cache: originalRequest.cache,
      mode: originalRequest.mode,
      redirect: originalRequest.redirect,
      referrer: originalRequest.referrer,
      referrerPolicy: originalRequest.referrerPolicy,
      integrity: originalRequest.integrity,
      keepalive: originalRequest.keepalive,
      signal: originalRequest.signal,
      window: originalRequest.window, // Only applicable in browser context
    };

    // Important: For GET or HEAD requests, the `body` property must not be set,
    // as it will cause a `fetch` error.
    if (proxyRequestInit.method === 'GET' || proxyRequestInit.method === 'HEAD') {
      delete proxyRequestInit.body;
    }

    const barewireTargetUrl = config.proxyUrl;

    try {
      if (config.debug) {
        console.log(`[BarewireInterceptor] Forwarding to proxy: ${barewireTargetUrl} (original: ${requestUrl})`);
        // console.log('[BarewireInterceptor] Proxy Request Init:', proxyRequestInit); // Can be very verbose
      }
      // Use the original (un-intercepted) fetch function to make the request to the Barewire proxy.
      const response = await originalFetch!(barewireTargetUrl, proxyRequestInit);

      if (config.debug) {
        console.log(`[BarewireInterceptor] Proxy responded for ${requestUrl}: ${response.status}`);
      }

      return response;
    } catch (error) {
      console.error(`[BarewireInterceptor] Error routing request to Barewire proxy for ${requestUrl}:`, error);
      throw error; // Re-throw the error to indicate failure to the caller
    }
  };

  // Return the cleanup function, allowing the consumer to restore original fetch later.
  return restoreBarewireInterceptor;
}

/**
 * Restores the original fetch behavior, undoing the Barewire interception.
 * This should be called when the Barewire proxy is no longer needed to prevent memory leaks
 * or unexpected behavior.
 */
export function restoreBarewireInterceptor(): void {
  if (originalFetch !== null) {
    window.fetch = originalFetch;
    originalFetch = null;
    currentBarewireConfig = null; // Clear the stored configuration
    console.log('[BarewireInterceptor] Original fetch restored.');
  } else {
    console.warn('[BarewireInterceptor] No active interception to restore.');
  }
}