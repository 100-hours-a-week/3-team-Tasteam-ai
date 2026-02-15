{
  "state": "Done",
  "series": [],
  "annotations": [],
  "request": {
    "app": "dashboard",
    "requestId": "Q9335",
    "timezone": "browser",
    "panelId": 2,
    "dashboardId": 2,
    "dashboardUID": "prometheus-overview",
    "publicDashboardAccessToken": "",
    "range": {
      "from": "2026-02-11T04:19:51.585Z",
      "to": "2026-02-11T10:19:51.585Z",
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
        "expr": "sum by (pipeline) (rate(analysis_requests_total{job=\"fastapi\",status=\"success\"}[5m])) / (sum by (pipeline) (rate(analysis_requests_total{job=\"fastapi\"}[5m])) + 1e-9)",
        "legendFormat": "{{pipeline}}",
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
    "startTime": 1770805191585,
    "rangeRaw": {
      "from": "now-6h",
      "to": "now"
    },
    "endTime": 1770805191621
  },
  "timeRange": {
    "from": "2026-02-11T04:19:51.585Z",
    "to": "2026-02-11T10:19:51.585Z",
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