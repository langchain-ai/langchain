/**
 * OpenStreetMap embed (same approach as ui-playground headless-tools/components.tsx).
 */

type LocationMapProps = {
  latitude: number;
  longitude: number;
  accuracy?: number;
};

export function LocationMap({ latitude, longitude, accuracy }: LocationMapProps) {
  const delta = 0.005;
  const bbox = `${longitude - delta},${latitude - delta},${longitude + delta},${latitude + delta}`;
  const src = `https://www.openstreetmap.org/export/embed.html?bbox=${bbox}&layer=mapnik&marker=${latitude},${longitude}`;
  const externalHref = `https://www.openstreetmap.org/?mlat=${latitude}&mlon=${longitude}#map=16/${latitude}/${longitude}`;

  return (
    <div className="location-map">
      <div className="location-map-frame">
        <iframe
          src={src}
          title="Your location on OpenStreetMap"
          loading="lazy"
          referrerPolicy="no-referrer"
        />
      </div>
      <div className="location-map-meta">
        <span className="location-map-coords">
          {latitude.toFixed(5)}, {longitude.toFixed(5)}
          {accuracy !== undefined ? (
            <span className="location-map-acc"> ±{Math.round(accuracy)} m</span>
          ) : null}
        </span>
        <a href={externalHref} target="_blank" rel="noopener noreferrer" className="location-map-link">
          Open in OSM ↗
        </a>
      </div>
    </div>
  );
}
