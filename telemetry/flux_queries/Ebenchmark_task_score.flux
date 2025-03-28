from(bucket: "tplr")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "Ebenchmark_task")
  |> filter(fn: (r) => r._field == "score")
  |> filter(fn: (r) => r["uid"] =~ /${uid:regex}/)
  |> filter(fn: (r) => r["role"] == "evaluator")
  |> filter(fn: (r) => r["version"] =~ /${version:regex}/)
  |> group(columns: ["uid", "_field", "task"])
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
  |> yield(name: "mean")
