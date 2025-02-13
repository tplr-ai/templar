
CREATE VIEW v_version AS
SELECT 
    version,
    TO_CHAR(created_at, 'Mon DD, YYYY, HH12:MI AM +00:00') AS created_at,
    (EXTRACT(DAY FROM (NOW() - created_at)) || 'd ' ||
     EXTRACT(HOUR FROM (NOW() - created_at)) || 'h ' ||
     EXTRACT(MINUTE FROM (NOW() - created_at)) || 'min ' ||
     FLOOR(EXTRACT(SECOND FROM (NOW() - created_at))) || 's') AS up_time
FROM tbL_version ORDER BY id desc LIMIT 1;

CREATE VIEW v_validator_eval_info AS
SELECT 
    window_time, 
    loss_before, 
    loss_after, 
    loss_improvement, 
    loss_random_before, 
    loss_random_after, 
    loss_random_improvement, 
    mean_scores, 
    mean_moving_avg_scores,
    eval_uids::INTEGER eval_uids
FROM tbl_validator_eval_info aa
JOIN tbl_window_info bb ON aa.window_id = bb.id
JOIN tbl_version cc ON bb.id >= cc.window_id and cc.is_running = true
ORDER BY window_time ASC;

CREATE VIEW v_active_miners AS
SELECT 
    bb.window_time, 
    COALESCE(array_length(string_to_array(aa.active_miners, ','), 1), 0) AS active_miners_count, 
    COALESCE(array_length(string_to_array(aa.error_miners, ','), 1), 0) AS error_miners_count,
    COALESCE(array_length(string_to_array(aa.bad_miners, ','), 1), 0) AS bad_miners_count,
    COALESCE(array_length(string_to_array(aa.diff_miners, ','), 1), 0) AS diff_miners_count
FROM tbl_active_miners aa
JOIN tbl_window_info bb ON aa.window_id = bb.id
JOIN tbl_version cc ON bb.id >= cc.window_id and cc.is_running = true
ORDER BY bb.window_time ASC;

CREATE VIEW v_gradients AS
SELECT 
    window_time, 
    COUNT(neuron_id) AS gradients_count 
FROM tbl_gradients aa
JOIN tbl_window_info bb ON aa.window_id = bb.id
JOIN tbl_version cc ON bb.id >= cc.window_id and cc.is_running = true
GROUP BY window_time
ORDER BY window_time ASC;

CREATE VIEW v_overview AS
WITH aa AS (
    SELECT MAX(id) AS maxid FROM tbl_window_info
)
SELECT 
    bb.window_number, 
    sync_window_number, 
    floor(avg_window_duration * 10)/10 || 's' avg_window_duration, 
    gradient_retention,
	blocks_per_window,
	gather_miners,
	diff_miners
FROM aa
JOIN tbl_window_info bb ON aa.maxid = bb.id
JOIN tbl_run_metadata cc ON aa.maxid = cc.window_id
JOIN tbl_active_miners dd ON aa.maxid = dd.window_id;

CREATE VIEW v_eval_info_detail AS
SELECT 
    window_time, 
    miner_id, 
    score, 
    moving_avg_score, 
    weight 
FROM tbl_eval_info_detail aa
JOIN tbl_window_info bb ON aa.window_id = bb.id
JOIN tbl_version cc ON bb.id >= cc.window_id and cc.is_running = true
WHERE aa.vali_id = 1
ORDER BY window_time, miner_id ASC;

CREATE VIEW v_eval_info_detail_current AS
WITH aa AS (
    SELECT MAX(id) AS maxid FROM tbl_window_info
)
SELECT 
	miner_id::TEXT miner_id,
    moving_avg_score, 
    weight
FROM aa
JOIN tbl_window_info bb ON aa.maxid = bb.id
JOIN tbl_eval_info_detail cc ON aa.maxid = cc.window_id
ORDER BY cc.miner_id;

CREATE VIEW v_window_info AS
SELECT
  window_time,
  learning_rate
FROM
  tbl_window_info aa
JOIN tbl_version bb ON aa.id >= bb.window_id and bb.is_running = true
ORDER BY
  window_time DESC

SELECT   'UID' || miner_id::TEXT miner_id, sum(score) score , window_time 
FROM tbl_eval_info_detail aa
JOIN tbl_window_info bb ON aa.window_id = bb.id
JOIN tbl_version cc ON bb.id >= cc.window_id and cc.is_running = true
WHERE aa.vali_id = 1
GROUP BY window_time, miner_id  HAVING sum(score) > 0 

SELECT   'UID' || miner_id::TEXT miner_id, sum(moving_avg_score) score , window_time 
FROM tbl_eval_info_detail aa
JOIN tbl_window_info bb ON aa.window_id = bb.id
JOIN tbl_version cc ON bb.id >= cc.window_id and cc.is_running = true
WHERE aa.vali_id = 1
GROUP BY window_time, miner_id HAVING sum(moving_avg_score) > 0;


SELECT   'UID' || miner_id::TEXT miner_id, sum(weight) score , window_time 
FROM tbl_eval_info_detail aa
JOIN tbl_window_info bb ON aa.window_id = bb.id
JOIN tbl_version cc ON bb.id >= cc.window_id and cc.is_running = true
WHERE aa.vali_id = 1
GROUP BY window_time, miner_id  HAVING sum(weight) > 0;
