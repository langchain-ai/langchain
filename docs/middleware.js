import { next } from "@vercel/edge";
import redirects from "./redirects.json";

export default function middleware(request) {
  const url = new URL(request.url);
  const pathname = url.pathname;

  // Find matching redirect
  const redirect = redirects.find((r) => {
    const source = r.source.replace("(/?)", "");
    return pathname === source || pathname === source + "/";
  });

  if (redirect) {
    return Response.redirect(redirect.destination, 308);
  }

  return next();
}
