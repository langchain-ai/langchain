import React from 'react';
import Translate, {translate} from '@docusaurus/Translate';
import {PageMetadata} from '@docusaurus/theme-common';
import Layout from '@theme/Layout';
import suggestedLinks from './removed_pages_v01.json'

import {useLocation} from 'react-router-dom';

function LegacyBadge() {
  return (
    <span className="badge badge--secondary">LEGACY</span>
  );
}

export default function NotFound() {
  const location = useLocation();
  const pathname = `${location.pathname}/`;
  const {canonical, alternative} = suggestedLinks[pathname] || {};

  return (
    <>
      <PageMetadata
        title={translate({
          id: 'theme.NotFound.title',
          message: 'Page Not Found',
        })}
      />
      <Layout>
        <main className="container margin-vert--xl">
          <div className="row">
            <div className="col col--6 col--offset-3">
              <h1 className="hero__title">
                  {canonical ? 'Page Moved' : alternative ? 'Page Removed' : 'Page Not Found'}
              </h1>
              {
                canonical ? (
                  <h3>You can find the new location <a href={canonical}>here</a>.</h3>
                ) : alternative ? (
                  <p>The page you were looking for has been removed.</p>
                ) : (
                  <p>We could not find what you were looking for.</p>
                )
              }
              {alternative && (
                <p>
                  <details>
                    <summary>Alternative pages</summary>
                      <ul>
                        {alternative.map((alt, index) => (
                          <li key={index}>
                            <a href={alt}>{alt}</a>{alt.startsWith('/v0.1/') && <>{' '}<LegacyBadge/></>}
                          </li>
                        ))}
                      </ul>
                  </details>
                </p>
              )}
              <p>
                  Please contact the owner of the site that linked you to the
                  original URL and let them know their link {canonical ? 'has moved.' : alternative ? 'has been removed.' : 'is broken.'}
              </p>
            </div>
          </div>
        </main>
      </Layout>
    </>
  );
}
