-- Player churn: one row per player, label + first-24-hour features.
-- Source: Google's public Firebase sample for the game "Flood It!".
-- Runs in the free BigQuery sandbox.

WITH events AS (
  SELECT
    user_pseudo_id,
    event_name,
    event_timestamp,
    TIMESTAMP_MICROS(event_timestamp) AS event_ts,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'engagement_time_msec') AS engagement_time_msec,
    geo.country,
    device.operating_system,
    device.language
  FROM `firebase-public-project.analytics_153293282.events_*`
),

players AS (
  SELECT
    user_pseudo_id,
    MIN(event_ts) AS first_seen,
    MAX(event_ts) AS last_seen
  FROM events
  GROUP BY user_pseudo_id
),

labelled AS (
  SELECT
    user_pseudo_id,
    first_seen,
    last_seen,
    -- bounced: gone within 10 minutes of the first event; excluded below
    IF(last_seen <= TIMESTAMP_ADD(first_seen, INTERVAL 10 MINUTE), 1, 0) AS bounced,
    -- churned: no activity after the first 24 hours
    IF(last_seen < TIMESTAMP_ADD(first_seen, INTERVAL 24 HOUR), 1, 0) AS churned
  FROM players
),

context AS (
  -- country / OS / language as seen on the player's first event
  SELECT
    user_pseudo_id,
    ARRAY_AGG(STRUCT(country, operating_system, language) ORDER BY event_timestamp LIMIT 1)[OFFSET(0)] AS c
  FROM events
  GROUP BY user_pseudo_id
),

first_day AS (
  -- behaviour in the first 24 hours only, so the model never sees the future
  SELECT
    e.user_pseudo_id,
    COUNTIF(e.event_name = 'session_start')            AS sessions,
    SUM(IFNULL(e.engagement_time_msec, 0)) / 1000      AS engagement_seconds,
    COUNTIF(e.event_name = 'screen_view')              AS screen_views,
    COUNTIF(e.event_name LIKE 'level_start%')          AS levels_started,
    COUNTIF(e.event_name LIKE 'level_complete%')       AS levels_completed,
    COUNTIF(e.event_name LIKE 'level_fail%')           AS levels_failed,
    COUNTIF(e.event_name LIKE 'level_reset%')          AS levels_reset,
    COUNTIF(e.event_name = 'post_score')               AS scores_posted,
    COUNTIF(e.event_name = 'select_content')           AS content_selected
  FROM events e
  JOIN players p USING (user_pseudo_id)
  WHERE e.event_ts < TIMESTAMP_ADD(p.first_seen, INTERVAL 24 HOUR)
  GROUP BY e.user_pseudo_id
)

SELECT
  l.user_pseudo_id,
  l.churned,
  c.c.country,
  c.c.operating_system,
  c.c.language,
  EXTRACT(DAYOFWEEK FROM l.first_seen) AS first_day_of_week,
  EXTRACT(HOUR FROM l.first_seen)      AS first_hour,
  f.* EXCEPT (user_pseudo_id)
FROM labelled l
JOIN context c USING (user_pseudo_id)
JOIN first_day f USING (user_pseudo_id)
WHERE l.bounced = 0
