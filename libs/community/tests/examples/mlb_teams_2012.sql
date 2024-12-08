-- Provisioning table "mlb_teams_2012".
--
-- psql postgresql://postgres@localhost < mlb_teams_2012.sql
-- crash < mlb_teams_2012.sql

DROP TABLE IF EXISTS mlb_teams_2012;
CREATE TABLE mlb_teams_2012 ("Team" VARCHAR, "Payroll (millions)" FLOAT, "Wins" BIGINT);
INSERT INTO mlb_teams_2012
    ("Team", "Payroll (millions)", "Wins")
VALUES
    ('Nationals', 81.34, 98),
    ('Reds', 82.20, 97),
    ('Yankees', 197.96, 95),
    ('Giants',       117.62, 94),
    ('Braves',        83.31, 94),
    ('Athletics',     55.37, 94),
    ('Rangers',      120.51, 93),
    ('Orioles',       81.43, 93),
    ('Rays',          64.17, 90),
    ('Angels',       154.49, 89),
    ('Tigers',       132.30, 88),
    ('Cardinals',    110.30, 88),
    ('Dodgers',       95.14, 86),
    ('White Sox',     96.92, 85),
    ('Brewers',       97.65, 83),
    ('Phillies',     174.54, 81),
    ('Diamondbacks',  74.28, 81),
    ('Pirates',       63.43, 79),
    ('Padres',        55.24, 76),
    ('Mariners',      81.97, 75),
    ('Mets',          93.35, 74),
    ('Blue Jays',     75.48, 73),
    ('Royals',        60.91, 72),
    ('Marlins',      118.07, 69),
    ('Red Sox',      173.18, 69),
    ('Indians',       78.43, 68),
    ('Twins',         94.08, 66),
    ('Rockies',       78.06, 64),
    ('Cubs',          88.19, 61),
    ('Astros',        60.65, 55)
;
