# Design — "¿Cuánto tarda un modelo en aprender un idioma nuevo?" (Experimento de cifrados)

**Fecha:** 2026-07-15
**Estado:** Diseño aprobado en brainstorming (pendiente review del spec por el usuario)
**Tipo:** Parte 2 de la serie "bordes del lenguaje" (javieraguilar.ai). Parte 1 = repetición (publicada). Parte 3 (futura) = multimodal.
**Repo:** `llm-language-limits` (público), nuevo subdirectorio `experiments/ciphers/`. Reutiliza el harness de la parte 1 (`src/llm_language_limits/`: clients, runner, cloud sweep, storage).

---

## 1. Motivación y tesis

Pregunta: **¿cuán rápido infiere un modelo que le hablas en un código no-humano, y cuándo empieza a hablarlo él?** No cuánto *sabe* descifrar si se lo explicas (eso es ejecución), sino cuánto tarda en *pillarlo* por exposición y en *adoptarlo*.

Ejes:
- **Capacidad** — qué modelos infieren qué cifrados (¿los pequeños hacen suelo? no se asume; se mide).
- **Latencia** — dos curvas: **comprensión** (cuándo decodifica/actúa bien) y **producción** (cuándo responde consistentemente en el código).
- **Dificultad** — una escalera de cifrados; su orden real es parte del hallazgo.

Gancho práctico: **robustez** — los cifrados como vector de *bypass* de guardarraíles (à la CipherChat). Espejo humano: **criptoanálisis** (análisis de frecuencias, el momento "ajá"). Tono: honesto/matizado, como la parte 1.

## 2. Diseño experimental

- **IV1 — Cifrado × dificultad (~8-10):** organizados por tipo de transformación, de trivial a durísimo:
  1. *Sustitución carácter→carácter:* ROT13, sustitución aleatoria (permutación arbitraria del alfabeto — requiere análisis de frecuencias).
  2. *Mapeo a otros símbolos:* letras→dígitos (a=1…), Morse.
  3. *Codificación base:* binario, base64.
  4. *Transposición:* inversión total del texto, permutación por bloques (con clave).
  5. *Mezcla de alfabetos:* homóglifos latino↔cirílico.
  6. *Estructural/con pérdida:* disemvoweling (quitar vocales).
  El orden de dificultad es hipótesis a testar, no dado.

- **IV2 — Protocolo (los 3, comparados; podar si 2-3 redundan):**
  1. **Inferencia pura:** hablamos en el código desde el turno 1, sin explicación ni ejemplos etiquetados.
  2. **Few-shot (piedra Rosetta):** k pares (plano ↔ codificado) antes de conversar; se barre k.
  3. **Escalado:** empieza en inferencia pura; si no lo pilla en X turnos, se añaden pistas graduales (ejemplos → nombre del cifrado → "responde en el mismo código").

- **IV3 — Turnos:** conversación multi-turno, una tarea codificada por turno, hasta un tope (a calibrar en smoke; p. ej. 15-30). La **latencia = índice de turno** del hito. Réplicas ≥3.

- **Modelos:** **piloto amplio** = Claude Opus + Claude Sonnet + GPT-5 (Azure) + Qwen2.5-7B Instruct + Qwen2.5-7B Base. Tras el piloto se **poda el roster según quién muestre señal** (no se asume que los pequeños hagan suelo). Reutiliza los clients de la parte 1.

## 3. Estímulos y oráculo (determinista — sin juez LLM para lo central)

Ventaja clave sobre la parte 1: **nosotros controlamos los cifrados**, así que codificar/decodificar es determinista y la métrica central es programática (verificable), no un juicio LLM.

- **Banco de tareas verificables cortas** (manifiesto versionado): instrucciones con respuesta comprobable programáticamente — p. ej. "responde con la palabra BANANA", "¿cuánto es 7+5?", "nombra la capital de Francia", "responde SÍ o NO: ¿el cielo es azul?". Variadas para no memorizar.
- **Cada cifrado = par `encode`/`decode`** puro y testeado (controlamos ambos sentidos).
- Cada turno: se toma una tarea del banco, se **codifica** con el cifrado activo, se envía.

**Comprensión (dos señales, ambas):**
- (a) *Acción* — el modelo ACTÚA bien sobre la tarea decodificada (su respuesta satisface el oráculo de esa tarea), responda en el idioma que responda. Independiente del idioma de salida.
- (b) *Decode explícito* — en turnos marcados se le pide reescribir el mensaje a texto plano; se comprueba con match tolerante (normalización; un juez LLM ligero SOLO para el fuzzy-match si hace falta, no para la métrica de acción).

**Producción:**
- Se **decodifica la salida del modelo** con el cifrado inverso; ¿es texto válido/coherente en el código? Latencia = primer turno con salida válida en código. **Consistencia/recaída** = fracción de turnos posteriores que siguen en código vs. vuelven al inglés.
- Matiz por protocolo: en **inferencia pura**, producir en código es *emergente* (¿espeja el código sin que se lo pidan?); en **few-shot/escalado** está inducido. El contraste es dato.

**Métricas derivadas por celda (cifrado × protocolo × modelo × réplica):** turno-de-primera-comprensión (acción), turno-de-primer-decode-explícito-correcto, turno-de-primera-producción-válida, tasa de consistencia de producción, precisión de decode a turno-fijo (para el ranking).

## 4. Reutilización + ejecución

- Sobre el harness de la parte 1: `src/llm_language_limits/` (clients Anthropic/Azure/Modal, `run_matrix`, rate limiter, cloud sweep in-cluster, storage JSONL/Parquet, resume). Código nuevo en `experiments/ciphers/` + un módulo `src/llm_language_limits/ciphers.py` (encode/decode + tests) y `oracle.py` (tareas verificables + chequeo).
- **Ejecución escalonada:** tests unitarios de cifrados (encode/decode round-trip) → smoke (1 modelo barato, 1-2 cifrados) → **piloto** (roster amplio, subconjunto de cifrados y protocolos → PODAR roster/cifrados/protocolos + calibrar tope de turnos) → barrido completo. Resume + estimación de coste por proveedor (Anthropic/Azure/Modal) antes de cada run.
- **Jailbreak/robustez:** diseñado como **extensión opcional, decidida TRAS el piloto** — si hay cifrados que los modelos comprenden con fiabilidad, valorar una sonda pequeña y acotada (peticiones rechazadas en claro, enviadas cifradas; ¿sube el cumplimiento?), como investigación de seguridad legítima y reportada con cuidado. No forma parte del alcance base.

## 5. Espejo humano y artículo
Sección conceptual: criptoanálisis humano (frecuencias, patrones, el "ajá"), y por qué "entender un código" no es lo mismo que "seguir instrucciones en él". Conecta con la parte 1 (bordes del lenguaje) y con el gancho de robustez.

## 6. Riesgos y decisiones abiertas
- **¿Los modelos hacen suelo?** Desconocido → el piloto decide el roster y qué cifrados retener (algunos pueden ser imposibles para todos, otros triviales para todos; interesan los del medio).
- **Tope de turnos** para no cortar una comprensión tardía ni malgastar; se calibra en smoke/piloto.
- **Coste:** conversaciones multi-turno × cifrados × protocolos × modelos; el multi-turno crece. Acotar en el piloto; Azure con cuota estrecha (lección de la parte 1).
- **Decode explícito fuzzy-match:** definir normalización; juez LLM ligero solo si el match programático es insuficiente.
- **Contaminación:** cifrados muy conocidos (ROT13, base64) los "sabe" de entrenamiento → no mide inferencia sino memoria. Incluir cifrados **novedosos/aleatorios** (sustitución aleatoria, permutación por bloques con clave) para separar inferencia real de reconocimiento.

## 7. Fuera de alcance (YAGNI)
- Parte 3 (multimodal / más allá de la percepción humana) — experimento aparte.
- Fine-tuning o entrenamiento; esto es solo inferencia en contexto.
- Interpretabilidad mecánica de cómo el modelo decodifica.
