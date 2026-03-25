# Agente LangGraph PDF (DEV-107)

## Objetivo

Este proyecto implementa un agente basado en LangGraph capaz de transformar una instruccion en lenguaje natural en un documento PDF final.

La solucion se organiza en dos pasos principales dentro del grafo:

1. `designer_node`: interpreta la peticion del usuario y genera un documento HTML5 completo con estilos CSS incrustados.
2. `pdf_generator_node`: toma el HTML generado y lo compila a un archivo PDF final utilizando `xhtml2pdf`.

De esta forma, el flujo separa con claridad la fase de diseno del contenido y la fase de renderizado del documento final.

## Arquitectura del Grafo

El estado compartido del agente se define mediante `AgentState`, un `TypedDict` con los siguientes campos:

- `messages`: historial de mensajes del flujo, gestionado con `add_messages`.
- `html_content`: contenido HTML generado por el nodo de diseno.
- `pdf_path`: ruta del archivo PDF resultante.

El grafo conecta los nodos en una secuencia lineal:

`START -> designer -> generator -> END`

### Componentes principales

- `designer_node`
  Recibe el ultimo mensaje del usuario, invoca el modelo `ChatGroq` y le pide actuar como un maquetador experto para producir un HTML5 completo con CSS embebido.

- `_extract_html`
  Funcion auxiliar que limpia la respuesta del LLM en caso de que devuelva el HTML dentro de bloques Markdown o con texto adicional. Su objetivo es garantizar que el nodo de generacion reciba un documento HTML valido.

- `pdf_generator_node`
  Usa `xhtml2pdf` para convertir el HTML almacenado en `html_content` en un archivo `salida.pdf`. Si la libreria detecta errores durante la generacion, se lanza una excepcion explicita.

## Codigo Final

```python
from __future__ import annotations

import re
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from xhtml2pdf import pisa


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    html_content: str
    pdf_path: str


def _extract_html(content: str) -> str:
    """Normaliza la salida del LLM para quedarnos solo con HTML valido."""
    cleaned = content.strip()

    code_block_match = re.search(
        r"```(?:html)?\s*(.*?)```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if code_block_match:
        cleaned = code_block_match.group(1).strip()

    html_start = cleaned.lower().find("<!doctype html")
    if html_start == -1:
        html_start = cleaned.lower().find("<html")

    if html_start > 0:
        cleaned = cleaned[html_start:].strip()

    return cleaned


def designer_node(state: AgentState) -> AgentState:
    """Genera un documento HTML5 completo con estilos incrustados."""
    if not state["messages"]:
        raise ValueError("El estado no contiene mensajes de entrada.")

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)

    system_prompt = (
        "Actua como un maquetador experto en documentos empresariales. "
        "A partir de la solicitud del usuario, genera un documento HTML5 completo, "
        "bien estructurado y visualmente cuidado, incluyendo CSS incrustado dentro "
        "de una etiqueta <style>. "
        "Devuelve solo HTML valido, sin explicaciones adicionales."
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            state["messages"][-1],
        ]
    )

    html_content = _extract_html(
        response.content if isinstance(response.content, str) else str(response.content)
    )

    if "<html" not in html_content.lower():
        raise ValueError("La respuesta del modelo no contiene un documento HTML valido.")

    return {
        "html_content": html_content,
        "messages": [response],
    }


def pdf_generator_node(state: AgentState) -> AgentState:
    """Convierte el HTML generado en un archivo PDF."""
    html_content = state.get("html_content", "").strip()
    if not html_content:
        raise ValueError("No hay contenido HTML en el estado para generar el PDF.")

    pdf_path = "salida.pdf"
    with open(pdf_path, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(html_content, dest=result_file)

    if pisa_status.err:
        raise RuntimeError("Hubo un error al generar el PDF con xhtml2pdf.")

    return {"pdf_path": pdf_path}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("designer", designer_node)
graph_builder.add_node("generator", pdf_generator_node)

graph_builder.add_edge(START, "designer")
graph_builder.add_edge("designer", "generator")
graph_builder.add_edge("generator", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    load_dotenv()

    prompt = "Genera un informe de ventas mensual con tablas y colores corporativos azules"

    initial_state: AgentState = {
        "messages": [HumanMessage(content=prompt)],
        "html_content": "",
        "pdf_path": "",
    }

    result = graph.invoke(initial_state)

    print("PDF generado en:", result["pdf_path"])
```

## Instrucciones de Ejecucion

1. Instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

2. Configurar el archivo `.env` con la clave de Groq:

```env
GROQ_API_KEY=tu_clave_aqui
```

3. Ejecutar el agente:

```bash
python main.py
```

4. Tras la ejecucion correcta, el sistema generara el archivo `salida.pdf` en la raiz del proyecto.

## Resultado Esperado

Al ejecutar el flujo, el agente recibira una instruccion en lenguaje natural, generara automaticamente una propuesta visual en HTML5 con CSS y la convertira en un PDF final listo para su uso o distribucion.
