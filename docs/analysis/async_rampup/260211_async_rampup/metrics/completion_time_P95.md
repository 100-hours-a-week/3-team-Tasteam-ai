{
  "state": "Done",
  "series": [],
  "annotations": [],
  "request": {
    "app": "dashboard",
    "requestId": "Q9111",
    "timezone": "browser",
    "panelId": 3,
    "dashboardId": 2,
    "dashboardUID": "prometheus-overview",
    "publicDashboardAccessToken": "",
    "range": {
      "from": "2026-02-11T04:15:40.750Z",
      "to": "2026-02-11T10:15:40.750Z",
      "raw": {
        "from": "now-6h",
        "to": "now"
      }
    },
    "timeInfo": "",
    "interval": "30s",
    "intervalMs": 30000,
    "targets": [
      {
        "datasource": {
          "type": "prometheus",
          "uid": "prometheus"
        },
        "expr": "histogram_quantile(0.95, sum by (le, pipeline, analysis_type) (rate(analysis_processing_time_seconds_bucket{job=\"fastapi\"}[5m])))",
        "legendFormat": "{{pipeline}} - {{analysis_type}}",
        "refId": "A"
      }
    ],
    "maxDataPoints": 708,
    "scopedVars": {
      "__interval": {
        "text": "30s",
        "value": "30s"
      },
      "__interval_ms": {
        "text": "30000",
        "value": 30000
      }
    },
    "startTime": 1770804940750,
    "rangeRaw": {
      "from": "now-6h",
      "to": "now"
    },
    "endTime": 1770804940774
  },
  "timeRange": {
    "from": "2026-02-11T04:15:40.750Z",
    "to": "2026-02-11T10:15:40.750Z",
    "raw": {
      "from": "now-6h",
      "to": "now"
    }
  },
  "timings": {
    "dataProcessingTime": 0
  },
  "structureRev": 1
}