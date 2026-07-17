# AI Influencer OS
# Image Generation Guide

---

# Feed Images

Generate Latte

```bash
python3 src/main.py coffee_shop/latte
```

Generate Outdoor Cafe

```bash
python3 src/main.py coffee_shop/outdoor
```

Generate Reading

```bash
python3 src/main.py coffee_shop/reading
```

Generate Book

```bash
python3 src/main.py coffee_shop/book
```

---

# Instagram Story

Story 01

```bash
python3 src/main.py coffee_shop/window
```

Story 02

```bash
python3 src/main.py coffee_shop/counter
```

Story 03

```bash
python3 src/main.py coffee_shop/croissant
```

Story 04

```bash
python3 src/main.py coffee_shop/dessert
```

---

# Caption

Generate Feed Caption

```bash
python3 src/caption_main.py coffee_shop/latte
```

---

# Workflow

1. Update today's YAML
2. Generate Feed Image
3. Generate Story Images
4. Generate Caption
5. Review images
6. Publish to Instagram

---

# Folder Structure

08_content/

    image_requests/
        coffee_shop/

            latte.yaml

            window.yaml

            counter.yaml

            croissant.yaml

            dessert.yaml

09_production/

10_apps/

---

# Current Aspect Ratio

Feed

4:5

Story

9:16

---

# Persona

Current Persona

Aiko

Platform

Instagram

Style

Ultra Photorealistic Editorial Travel Photography