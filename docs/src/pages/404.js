import React from 'react';
import Layout from '@theme/Layout';

export default function NotFound() {
  return (
    <Layout title="Page Not Found">
      <div style={{ padding: '4rem', textAlign: 'center' }}>
        <h1>404 - Page Not Found</h1>
        <p>Oops! The page you're looking for doesn't exist.</p>
        <a href="/">Go back to the homepage</a>
      </div>
    </Layout>
  );
}
