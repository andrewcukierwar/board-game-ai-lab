# docker/ui.Dockerfile  ── Vite + React

# ---------- build stage ----------
FROM node:22 AS build
WORKDIR /app
COPY ui/package*.json ./
RUN npm ci
COPY ui/ ./
# Explicitly override any env-file value: Compose must use Nginx's /v1 proxy.
RUN VITE_API_BASE= npm run build

# ---------- serve stage ----------
FROM nginx:1.27-alpine AS runtime
COPY ui/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
