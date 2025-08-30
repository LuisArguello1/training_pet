# Guía para usar GitHub en el proyecto AgePet

---

## Pasos para subir un repositorio a GitHub

**Subir un repositorio a GitHub**
```bash
git init 
git add .
git commit -m "first commit"
```

**Consultar en qué rama te encuentras (importante)**
```bash
git status
```

**Si quieres crear una rama nueva puedes usar (opcional)**
```bash
git branch main # o master
```

**O cambiar de rama (opcional si estás en la rama correcta)**
```bash
git checkout main
```

**Decirle a GitHub dónde está el repositorio**
```bash
git remote add origin <url>
```

**Subir los cambios al repositorio**
```bash
git push origin <rama actual>
```

---

## Pasos para actualizar cambios en el mismo repositorio

```bash
git add .
git commit -m "<Describir los cambios hechos>"
git push origin <rama actual>
```

---

## Pasos para clonar un repositorio de GitHub 

```bash
git clone <url>
```

---

## Cambiar el remote de un repositorio

```bash
git init
git add .
git commit -m "<Describir brevemente cambios hechos>"
git remote set-url origin <nueva_url_del_repositorio>
git push origin <rama actual>
```

---

## Trabajo colaborativo en GitHub

**El líder crea y sube el repositorio a GitHub.**

**El líder agrega a los integrantes como colaboradores:**
- Ir a "Settings" > "Collaborators" en el repositorio.
- Agregar los nombres de usuario de los integrantes.

**Cada integrante clona el repositorio en su computadora**
```bash
git clone <url_del_repositorio>
```

**Cada integrante crea una rama para su tarea o funcionalidad**
```bash
git checkout main
git pull origin main
git checkout -b nombre-de-la-rama
```
Ejemplo: `interfaz-villavicencio`, `backend-santiago`

**Trabajar en la rama creada**
- Realizar cambios en los archivos.
- Guardar y agregar los cambios:
  ```bash
  git add .
  git commit -m "Descripción de los cambios"
  ```

**Importante**
- Crear un Pull Request en GitHub para fusionar la rama con `main`.
- El líder revisa y aprueba el Pull Request.

**Actualizar la rama local antes de seguir trabajando**
```bash
git checkout main
git pull origin main
git checkout nombre-de-la-rama
git merge main
```

---

> Esto es una guía de cómo usar GitHub, deben de aprendérselo porque vamos a estar usando esto frecuentemente.
