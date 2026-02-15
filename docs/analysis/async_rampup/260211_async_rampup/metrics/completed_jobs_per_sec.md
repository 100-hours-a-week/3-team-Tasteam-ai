{
  "state": "Done",
  "series": [],
  "annotations": [],
  "request": {
    "app": "dashboard",
    "requestId": "Q9157",
    "timezone": "browser",
    "panelId": 4,
    "dashboardId": 2,
    "dashboardUID": "prometheus-overview",
    "publicDashboardAccessToken": "",
    "range": {
      "from": "2026-02-11T04:16:31.555Z",
      "to": "2026-02-11T10:16:31.555Z",
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
        "expr": "sum by (pipeline, analysis_type) (rate(analysis_requests_total{job=\"fastapi\",status=\"success\"}[5m]))",
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
    "startTime": 1770804991555,
    "rangeRaw": {
      "from": "now-6h",
      "to": "now"
    },
    "endTime": 1770804991590
  },
  "timeRange": {
    "from": "2026-02-11T04:16:31.555Z",
    "to": "2026-02-11T10:16:31.555Z",
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