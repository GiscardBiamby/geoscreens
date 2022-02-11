curl -X GET https://localhost:8080/api/projects/{id}/export?exportType=JSON &
download_all_tasks=true -H 'Authorization: Token abc123' --output 'annotations.json'
curl -X GET http://localhost:8093/api/projects/2/export?exportType=JSON &
download_all_tasks=true -H 'Authorization: Token 3ac2082c83061cf1056d636a25bee65771792731' --output 'annotations.json'

curl http://localhost:8093/api/projects -H "Authorization: Token 3ac2082c83061cf1056d636a25bee65771792731"
curl http://localhost:8093/api/projects/2 -H "Authorization: Token 3ac2082c83061cf1056d636a25bee65771792731" --output 'annotations.json'

# Export all tasks from label-studio
curl "http://localhost:8093/api/projects/5/export?exportType=JSON&download_all_tasks=true" -H "Authorization: Token 3ac2082c83061cf1056d636a25bee65771792731" --output '/shared/gbiamby/geo/exports/geoscreens_005_tasks_export.json'


curl "http://localhost:6008/api/projects/5/export/formats" -H "Authorization: Token 3ac2082c83061cf1056d636a25bee65771792731"