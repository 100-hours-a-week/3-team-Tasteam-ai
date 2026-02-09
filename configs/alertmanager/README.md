# Alertmanager 설정

알람이 **killfog1@gmail.com** 으로 이메일로 발송되도록 설정되어 있습니다.

## Gmail 앱 비밀번호 설정

1. Gmail에서 [2단계 인증](https://myaccount.google.com/signinoptions/two-step-verification)을 켠 뒤
2. [앱 비밀번호](https://myaccount.google.com/apppasswords)에서 "앱 비밀번호"를 생성합니다.
3. `configs/alertmanager/alertmanager.yml` 의 `smtp_auth_password: "YOUR_GMAIL_APP_PASSWORD"` 부분을 생성한 16자리 앱 비밀번호로 교체합니다.

비밀번호를 저장소에 넣고 싶지 않다면, 이 파일을 로컬에서만 수정하고 커밋하지 않거나, `alertmanager.yml` 을 `.gitignore` 에 추가한 뒤 로컬 복사본만 사용할 수 있습니다.
