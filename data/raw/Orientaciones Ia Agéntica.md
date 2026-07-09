# **INTELIGENCIA ARTIFICIAL AGÉNTICA DESDE LA PERSPECTIVA DE PROTECCIÓN DE DATOS**

V1.2 de febrero de 2026

## **RESUMEN EJECUTIVO**

Un agente de IA es un sistema de inteligencia artificial que utiliza modelos de lenguaje para cumplir un objetivo. Estas orientaciones son una introducción a las cuestiones de protección de datos que pueden surgir cuando responsables y encargados de tratamiento decidan utilizar sistemas de IA agéntica para implementar tratamientos de datos personales.

El objeto de este documento no es analizar el cumplimiento de un tratamiento concreto que emplea agentes de IA, sino como gestionar las peculiaridades que se incorporan en un tratamiento por el hecho de implementarse total o parcialmente con agentes.

Conocer esta tecnología es clave para adoptar decisiones informadas y basadas en evidencia sobre su implementación en tratamientos de datos personales. No basta el conocimiento como usuario: es necesario comprender sus fundamentos, alcances, límites y la manera en que se aplica. Tanto el rechazo irracional de la IA agéntica como su aceptación acrítica en el tratamiento de datos personales pueden resultar perjudiciales. En particular, hay que aprovechar de forma proactiva las oportunidades que ofrece esta tecnología para una mayor protección de datos desde el diseño y como herramienta PET por sí misma.

El texto se estructura realizando inicialmente una breve descripción de qué son los sistemas IA agénticos. A continuación, se analizarán las posibles vulnerabilidades de estos sistemas que afectan a la protección de datos, los aspectos de cumplimiento de la normativa de protección de datos y las amenazas específicas que pueden aprovechar las distintas vulnerabilidades. Finalmente, el documento enumera medidas que podría adoptar un responsable o encargado para garantizar el cumplimiento de la normativa de protección de datos y reducir o eliminar los impactos que la IA agéntica presenta en su despliegue en tratamientos con relación a los derechos y libertades de los sujetos de los datos. Estos análisis se centrarán en lo que es más distintivo en la IA agéntica como sistema en un tratamiento de datos personales, más allá de las vulnerabilidades, amenazas y medidas que son bien conocidas de las inteligencias artificiales generativas, o de otros elementos que componen estos sistemas.

Palabras clave: Internet y nuevas tecnologías, machine learning, aprendizaje automático, inteligencia artificial, protección de datos desde el diseño y por defecto, decisiones automatizadas.

## **I. INTRODUCCIÓN**

La automatización de tareas es el uso de tecnologías para ejecutar actividades repetitivas sin intervención humana constante. Este enfoque transforma eficientemente procesos que antes se realizaban completamente de forma manual, liberando tiempo para tareas de mayor valor. La utilización de sistemas de automatización se puede encontrar desde entornos industriales a entornos de oficina, incluyendo cualquier otro sector productivo o de servicios.

El desarrollo de los grandes modelos de lenguaje (LLMs por sus siglas en inglés) cambia completamente el paradigma de la automatización haciendo surgir el concepto de IA agéntica como sistemas basados en IA con capacidad de actuar de forma autónoma para conseguir el cumplimiento de objetivos: los agentes de IA. La integración de modelos de lenguaje supone un salto cualitativo en la eficacia y complejidad de las tareas que pueden llevar a cabo, lo que abre un universo de posibilidades para la mejora de los procesos empresariales y en las Administraciones Públicas. A su vez, el uso de sistemas que implementan el paradigma de la AI agéntica (agentes de IA) trabajando colaborativamente para automatizar múltiples procesos conlleva un cambio en la propia concepción de la implementación de los procesos, flujos de trabajo o workflows de las entidades, así como del uso de la inteligencia artificial generativa (en adelante IAG) en el entorno laboral.

La capacidad que tienen los sistemas de IA agéntica para operar con autonomía, enriquecerse con la información del entorno digital y ejecutar tareas complejas, introduce nuevos retos en muchos aspectos, entre ellos, en el ámbito laboral, la gestión y control de la organización, de resiliencia, de seguridad (safety) y ciberseguridad, aspectos éticos, la posibilidad de fraude, sobre la imagen corporativa, etc., además de los relacionados con la protección de datos personales. También, como sistemas de inteligencia artificial en sí mismos y por el tratamiento de datos que permiten implementar, pueden surgir obligaciones que se deriven de normas generales, como del Reglamento de Inteligencia Artificial o el Reglamento de Datos, o de normas específicas por ámbito de aplicación.

En este documento se va a realizar una introducción sobre las cuestiones de protección de datos que pueden surgir cuando responsables y encargados de tratamiento decidan utilizar sistemas de IA agéntica para implementar tratamientos de datos personales.

Este documento no va a abordar el empleo de agentes de IA en el ámbito doméstico (aunque pueden existir también implicaciones de cumplimiento normativo), ni aspectos sobre el desarrollo o evolución de modelos de lenguaje. Tampoco se aborda la cuestión de agentes de IA de en una organización en los que no hay tratamientos de datos personales.

Los agentes de IA son medios, sistemas, que permiten implementar tratamientos de datos personales introduciendo una mayor automatización. Un mismo agente de IA puede ser utilizado para implementar operaciones en distintos tratamientos de datos personales. Por otro lado, un agente de IA puede ser una parte de las operaciones de un tratamiento que incluya, para implementar otras operaciones, el uso de otros sistemas o de operaciones realizadas por un operador humano.

El objeto de este documento no es analizar el cumplimiento de un tratamiento concreto que emplea agentes de IA, sino como gestionar las peculiaridades que se incorporan en un tratamiento por el hecho de implementarse total o parcialmente con agentes. Distintos tratamientos y distintos tipos de agentes implementados en dichos tratamientos podrían tener distintas implicaciones en protección de datos. En el análisis que se realiza en este documento se estudiarán estas implicaciones de forma genérica, teniendo en cuenta que no son inherentes, ni forman necesariamente parte de su naturaleza, a todos los agentes ni a todo uso de la IA agéntica.

## **II. AGENTES DE IA**

Los agentes digitales, basados en software tradicional y sistemas de control son previos a la aparición de la IA, sin embargo, sus funcionalidades eran limitadas en comparación con lo que se puede conseguir con la AI agéntica.

La IA agéntica supone mucho más que emplear LLMs. Comprender su funcionamiento resulta esencial para crear el clima de confianza, a través de la evidencia, de que se han dispuesto las medidas y garantías adecuadas que permita a responsables y encargados de tratamiento obtener lo mejor de esta opción tecnológica.

### **A. AGENTE DE IA**

Un agente de IA es un sistema de inteligencia artificial que utiliza modelos de lenguajes para cumplir un objetivo. Un agente de IA actúa de manera adecuada según sus circunstancias y sus objetivos, es flexible ante entornos y metas cambiantes, aprende de la experiencia y toma decisiones apropiadas dadas sus limitaciones perceptivas y computacionales. Para ello, descompone tareas complejas en subtareas, que se ejecutan de forma planificada creando una cadena de razonamiento, cada una de ellas implementada con distintas herramientas y que perciben el entorno mediante el acceso a servicios internos y externos.

Los agentes de IA se podrían definir por las siguientes características de forma general:

* Autonomía: poder operar sin intervención humana constante.  
* Percepción del entorno: procesan entradas en tiempo real mediante sensores, interfaces con aplicaciones (APIs), cámaras, etc.  
* Acción: además de generar salidas de texto, código o multimedia, pueden ejecutar acciones externas.  
* Proactividad: anticipan necesidades o problemas en lugar de solo reaccionar, pudiendo iniciar acciones por sí mismos.  
* Planificación y razonamiento: permiten planifican secuencias de acciones para cumplir metas específicas.  
* Memoria y Adaptabilidad: Permiten definir el contexto, acumular experiencias, ajustar comportamientos a las reacciones del usuario y mejorar iterativamente mediante retroalimentación.

### **B. LA CADENA DE RAZONAMIENTO**

La cadena de razonamiento, pipeline o procesamiento, es el proceso interno mediante el cual el agente descompone un problema en pasos lógicos sucesivos y encadenados, hasta llegar a una decisión o respuesta final. Esta cadena puede ser corta o, en los agentes que crecen en complejidad, muy larga con múltiples etapas. Cada una de estas etapas puede involucrar distintos sistemas, formatos y niveles de confianza.

La flexibilidad de la cadena de razonamiento puede variar desde un plan rígido codificado o máquinas de estado finito, hasta modelos conversacionales donde las decisiones dependen de interacciones y modelos de razonamiento.

En este último caso es cuando aparecen los LLMs como uno de los componentes nucleares de los agentes IA. Lo distintivo es utilizarlos para descomposición de tareas. Conocer la cadena de razonamiento permitirá conocer el ciclo de vida del dato, la fuente del dato, la fecha y hora exacta de extracción, cuándo, dónde y por quién se produce su transformación, y cuándo, dónde, por quién y con qué finalidad y legitimidad se carga en un repositorio, se usa o se descarga desde un entorno a otro repositorio.

### **C. PATRONES DE LOS AGENTES DE IA**

La arquitectura de los agentes, también llamados patrones (patterns), implementa un marco de razonamiento, es lo que permite planificar y ejecutar tareas complejas, combinando procesamiento del lenguaje natural, razonamiento simbólico, interacción con el entorno digital y planificación orientada a objetivos, lo que les confiere un grado de independencia operativa.

Los agentes de IA pueden automatizar tareas repetitivas de procesamiento de datos, analizar información para apoyar la toma de decisiones humanas o interactuar directamente con usuarios terceros y otros sistemas digitales. A diferencia de los LLM, que son reactivos a acciones del usuario, los agentes pueden ser proactivos, y utilizar llamadas a/de herramientas en segundo plano.

Aunque los agentes de inteligencia artificial operan de manera autónoma en sus procesos de toma de decisiones, dependen de metas y reglas previamente definidas por las personas.

### **D. MULTIAGENTE**

La arquitectura multiagente combina varios agentes, donde el comportamiento y las responsabilidades de cada uno están estrictamente definidos, comparten información y decisiones, y son capaces de colaborar, competir o negociar entre sí para alcanzar objetivos más elaborados.

Existen diversas aproximaciones a la IA multiagente: modelos centralizados, de ejecución secuencial de agentes, distribuidos o jerárquicos. Cada agente podrá tener un rango de acción y de autonomía distinto. Siempre se necesitará una capa de orquestación que coordine el ciclo de vida de los agentes, gestione dependencias, asigne roles a cada agente, establezca limitaciones por dominios y resuelva conflictos.

### **E. DETALLE DE LA ARQUITECTURA DE UN MULTIAGENTE**

La arquitectura general de un sistema de IA agéntica es un factor que se debe conocer para poder realizar un análisis de sus posibilidades, limitaciones y vulnerabilidades. Los componentes de un sistema de IA agéntica podrían ser:

* Una aplicación que gestiona el interfaz para realizar tareas para el usuario.  
* Uno o más agentes que implementarán distintos patrones de razonamiento y técnicas.  
* Uno o más modelos LLMs (locales o remotos) utilizados para el razonamiento, generación de contenido y gestión de memoria.  
* Los servicios, incluidos funciones integradas, herramientas locales y código de la aplicación.  
* Los interfaces para el acceso e interconexión con herramientas y servicios externos (Internet, sensores, etc.).  
* Almacenamiento externo para memoria persistente a largo y corto plazo.  
* Servicios de soporte que formen parte de la infraestructura del agente.

## **III. AGENTES DE IA EN LOS TRATAMIENTOS**

Los agentes de IA son medios que permiten implementar tratamientos de datos personales introduciendo una mayor automatización. Un mismo agente de IA puede ser utilizado para implementar operaciones en distintos tratamientos de datos personales. Por otro lado, un agente de IA puede ser una parte de las operaciones de un tratamiento que incluya, para implementar el resto de las operaciones, el uso de otros sistemas o de operaciones realizadas por un operador humano.

Los agentes de IA son medios utilizados en un tratamiento que conforman su naturaleza, y también pueden alterar el contexto, el ámbito y añadir fines adicionales, además de alterar los riesgos inherentes al mismo. En el caso de tratamientos preexistentes, incluir IA agéntica obligará a una revisión de cumplimiento de dicho tratamiento.

El objeto de este documento no es analizar el cumplimiento de un tratamiento concreto que emplea agentes de IA, sino sobre aspectos distintivos que podrían surgir con relación a protección de datos por el hecho de implementarse total o parcialmente con sistemas de IA agéntica. A la hora de realizar dicho análisis es importante evitar la "niebla tecnológica" que puede provocar el solo nombrar IA agéntica a la hora de implementar un tratamiento. Puede facilitar iniciar el análisis del cumplimiento el aterrizar en entidades o personas físicas las mismas operaciones que realice el agente y su relación con los servicios a los que accede para luego analizar los aspectos distintos que introduce la IA agéntica.

La IA agéntica puede implicar la interacción con numerosos servicios internos y externos, a través de Internet, lo que expondría a los datos personales en una cadena de procesamiento en la que intervendría no solo el responsable, sino múltiples entidades bajo las políticas de privacidad, de cookies, de condiciones de servicio y contractuales de cada herramienta de terceros.

## **IV. VULNERABILIDADES Y TRATAMIENTOS DE DATOS PERSONALES**

En este capítulo vamos a realizar un análisis preliminar de las vulnerabilidades más importantes que podrían surgir en un tratamiento por implementar operaciones con IA agéntica. En la potencia y versatilidad de la IA agéntica reside, como en todo sistema complejo, sus principales vulnerabilidades.

Un sistema de inteligencia artificial agéntica integra diversos componentes de software. Asimismo, incluye interfaces tanto internas como externas que interactúan con múltiples servicios. En consecuencia, todas las vulnerabilidades inherentes a cada uno de estos sistemas forman parte de las vulnerabilidades de la IA agéntica. La interacción entre los distintos componentes puede originar nuevas vulnerabilidades o amplificar las existentes.

### **A. INTERACCIÓN CON EL ENTORNO**

En el marco del tratamiento, la IA agéntica tiene la capacidad de interaccionar con el entorno para ejecutar todas o parte de las operaciones de tratamiento. La interacción puede limitarse a la propia organización, o se puede extender a servicios externos. Las salidas podrían ser transparentes para dichos usuarios o para el responsable del tratamiento, pero contener datos personales o poder revelar información personal sobre las personas sujetas a dicho tratamiento.

**Acceso a datos de la organización y del usuario**

Una de las funcionalidades comunes de la IA agéntica es el acceso a los servicios y datos internos con el propósito de enriquecer el contexto para la ejecución de tareas. Un acceso no controlado podría producir tratamientos masivos de datos que irían en contra del principio de minimización, de limitación del tratamiento y de exactitud de la información. Si todo o parte de los componentes de la IA agéntica se implementan por encargados de tratamiento, podría suponer la comunicación de datos a terceros más allá de las finalidades del tratamiento.

**Capacidad de percepción y acción externamente a la organización**

La interconexión a servicios de Internet permite la interacción de los agentes con el entorno exterior a la organización. La existencia de comunicaciones de datos bidireccionales con múltiples intervinientes, sin el necesario control de la entidad, puede incrementar en un grado notable las vulnerabilidades. En cuanto a la información local que se envía al exterior, una libertad excesiva en la invocación de herramientas que recogen información interna podría provocar una comunicación de información innecesaria. La conexión con el exterior no solo para ejecutar acciones, sino para obtener información podría estar utilizando fuentes inadecuadas, obsoletas o sesgadas.

### **B. INTEGRACIÓN DE SERVICIOS**

La IA agéntica se fundamenta en la integración de múltiples servicios. Formando parte de la IA agéntica, combinará el uso de al menos un modelo de lenguaje, herramientas de gestión de memoria y de ejecución de tareas.

**Gestión de servicios**

Incluso cuando los servicios proceden del mismo proveedor, la industria provoca la evolución independiente de cada uno de ellos, con términos y contratos no homogéneos. Esto implica una mayor complejidad en la gestión de herramientas. Todo ello puede implicar desafíos para el cumplimiento normativo de protección de datos, como por ejemplo la gestión de numerosos intervinientes, control de tratamientos adicionales, conservación de datos, ejercicio de derechos, problemas de exactitud, etc.

**Facilidad de desplegar servicios de IA Agéntica**

Existen servicios de IA agéntica que son fáciles de desplegar, intuitivos, con herramientas que permiten diseñar tareas y conectividad entre componentes de forma muy rápida. Esto conlleva a la tentación de que usuarios no cualificados queden deslumbrados por sus posibilidades y realicen despliegues fuera de la gobernanza y las políticas de información de la entidad. Introducir un sistema de IA agéntica en el tratamiento implica rediseñar un proceso de la organización en el que deberían intervenir los responsables funcionales, TIC y de calidad, además del DPD.

### **C. MEMORIA**

La memoria es una de las grandes ventajas de la IA agéntica y uno de sus elementos nucleares junto con los LLMs. La memoria en agentes de IA es la capacidad de almacenar y recordar contextos y experiencias pasadas para mejorar la toma de decisiones, la adaptación y el rendimiento. Hay dos tipos muy distintos de memoria en la IA agéntica: la memoria de trabajo que permite las funcionalidades de los agentes y la memoria de gestión formada por todos los registros o logs de operación.

**Memoria de trabajo**

Los agentes usan distintos tipos de memoria "de trabajo", a corto y a largo plazo. La memoria a largo plazo puede categorizarse como: Memoria semántica, memoria episódica y memoria procedimental. Desde el punto de vista de la implementación de un tratamiento, la información almacenada se podría clasificar en: Memoria de la organización, Memoria para cada tratamiento específico, Memoria para cada caso tratado y Memoria de la persona usuaria para el mismo tratamiento.

La memoria puede presentar vulnerabilidades con relación a protección de datos como:

* Relevancia: es necesario establecer políticas claras y eficaces de qué se ha de almacenar.  
* Coherencia y completitud del contexto: la información almacenada debe tener la calidad suficiente.  
* Conservación: la información almacenada ha de ser la mínima necesaria para la operación del agente.  
* Integridad: la información almacenada permite manipular el resultado de las inferencias.

**Memoria de gestión**

Hay que tener también en cuenta el impacto que puede tener la memoria en la sombra que es común en el uso de los sistemas digitales, como son los registros de actividad o log. La memoria almacenada en los registros puede ser una Medida de protección de datos desde el diseño, pero también puede suponer un Impacto crítico o un Riesgo en caso de que personas autorizadas no cumplan sus deberes de confidencialidad.

**Ejercicio de derechos**

En la medida en que la memoria del sistema de la IA agéntica almacena datos personales en el marco de uno o más tratamientos, deberá contemplar desde el diseño la capacidad de ejercer todos los derechos del RGPD, entre ellos, acceso, rectificación, supresión, limitación y oposición.

### **D. AUTONOMÍA**

La automatización agéntica implica que los agentes pueden actuar de manera autónoma, sin recibir instrucciones explícitas de un usuario humano. La autonomía es significativa en la actuación del agente de IA con el entorno.

El nivel de autonomía del agente en el tratamiento es una decisión de diseño del responsable. Desde el punto de vista de la protección de datos personales hay varios aspectos que podrían ser afectados por dicha autonomía, como el cumplimiento de minimización, la automatización de decisiones (Art. 22 RGPD), la reversibilidad de acciones de alto impacto, la supervisión humana y la transparencia.

**Transparencia y supervisión humana**

Los usuarios y desarrolladores pueden enfrentar dificultades para entender cómo algunos agentes de IA toman decisiones. Una falta de transparencia de los procesos de razonamiento internos puede generar una ilusión de fiabilidad. Se intensifica el sesgo de automatización, llevando a los usuarios a aceptar las decisiones del sistema sin un análisis crítico suficiente.

**Planificación de tareas e interacción entre agentes**

En la medida en que los agentes van a implementar tratamientos de datos personales en la organización hay que garantizar que todas las subtareas son las necesarias, solo las necesarias, y en el orden adecuado. Hay que tener en cuenta que un LLM no "razona", sino que extrae modelos de descomposición de tareas. La complejidad técnica puede dar lugar a inestabilidad en el comportamiento emergente.

**Comportamiento no repetible**

En sistemas más complejos, si no existe un control estricto sobre las fuentes de información, los servicios a los que se accede, sus versiones, la planificación de tareas, la memoria y los posibles comandos generados, no es posible anticipar la salida del sistema. La falta de reproducibilidad dificulta la depuración y la implementación de políticas de seguridad.

**Capacidad de actuar en nombre del usuario o de la organización**

Los agentes de IA deben ser capaces de solicitar y utilizar credenciales de usuario y credenciales técnicas. La concesión de permisos excesivos a los agentes constituye un factor crítico. La proliferación de identidades de máquina asociadas a agentes genera un volumen elevado de cuentas técnicas cuya gestión resulta compleja.

## **V. ASPECTOS DE CUMPLIMIENTOS DE LA NORMATIVA DE PROTECCIÓN DE DATOS**

Un responsable de un tratamiento de datos personales podría elegir, entre los medios que va a utilizar para implementar el tratamiento, uno (o varios) sistemas de IA agéntica. Cuando los agentes de IA se utilizan en operaciones de tratamientos podrían surgir cuestiones como:

* Aparición de más intervienes.  
* Mayor extensión en el tipo y categorías de datos.  
* Mayor tratamiento de datos de las personas usuarias que interaccionan con los agentes de IA.  
* Menor transparencia en los tratamientos.  
* Acciones automatizadas con efecto en los sujetos de los datos.

### **A. DETERMINACIÓN DE RESPONSABILIDADES DE TRATAMIENTO**

Responsable del tratamiento es quien, solo o junto con otros, determine los fines y medios del tratamiento, independientemente de la forma que tengan dichos medios, ya sean sistemas de IA agéntica u otros. El responsable tendrá la obligación de garantizar el cumplimiento normativo y gestionar los nuevos riesgos.

Cuando un agente de IA se ejecute de forma totalmente local no cabría más análisis de la responsabilidad. Sin embargo, en la mayor parte de los casos, los sistemas de IA agéntica accederán a servicios de terceros fuera de la organización para cumplir con su propósito.

Si el agente envía información no personal a servicios de terceros sin vinculación a una persona usuaria, no habría una relación de protección de datos. Pero si se envía información personal, dichos servicios actuarían como encargados de ese tratamiento o responsables dependiendo de la relación específica.

En aplicación del principio de responsabilidad proactiva, el responsable deberá diseñar y documentar los flujos de datos del tratamiento, identificando para cada uno de los sistemas intervinientes los terceros implicados e identificando su rol.

### **B. TRANSPARENCIA**

En el caso de que el uso de sistemas de IA agéntica en un tratamiento implique destinatarios adicionales de los datos a los previstos en el propio tratamiento, deberá informarse debidamente de su identidad. Igualmente, deberá informarse de cualquier modificación que, debido al empleo de sistemas de IA agéntica, afecte los plazos de conservación de los datos personales o la existencia de decisiones automatizadas adicionales.

### **C. LEGITIMACIÓN, MINIMIZACIÓN Y LEVANTAMIENTO DE PROHIBICIONES**

La inclusión de sistemas de IA agéntica en un tratamiento podría implicar tratamientos adicionales de datos, aunque no necesariamente. Si existen tratamientos adicionales, deberá tener su base legitimadora establecida y, en caso de categorías especiales de datos, una circunstancia que levante la prohibición.

La minimización de datos ha de contemplarse desde el diseño de los tratamientos y trasladarlo al diseño o configuración de los agentes. La limitación del tratamiento también hay que abordarla desde el diseño.

### **D. REGISTRO DE ACTIVIDADES DE TRATAMIENTO**

El registro de actividades de tratamiento (RAT) es una herramienta fundamental para la gestión de cumplimiento. Cuando en un tratamiento se decida sustituir medios tradicionales por automatización agéntica deberá realizarse una actualización del RAT. El RAT es recomendable integrarlo en el catálogo de procesos del sistema de control de calidad de la entidad.

### **E. EJERCICIO DE DERECHOS**

El hecho de emplear sistemas de IA agéntica en un tratamiento no debe suponer una merma en el ejercicio de derechos y se deben implementar las medidas necesarias para garantizarlos. La configuración de los sistemas de memoria y la IA agéntica ha de estar técnicamente capaz desde el diseño para permitir gestionar el ejercicio de derechos.

### **F. AUTOMATIZACIÓN DE LAS DECISIONES**

**Artículo 22 del RGPD**

La incorporación de sistemas de IA agéntica en un tratamiento puede implicar automatización, pero no siempre implicará decisiones automatizadas en el sentido del artículo 22 del RGPD. Pero en caso de que existan, deberá evaluarse las condiciones que la permiten (art.22.2 del RGPD) y las medidas que será necesario implementar.

**Otras acciones automatizadas**

El empleo de una IA agéntica en un tratamiento puede implicar riesgos sobre el tratamiento de datos de personas físicas que no entren en el ámbito del art.22 del RGPD, como por ejemplo enviar información confidencial automáticamente.

### **G. GESTIÓN DEL RIESGO**

La gestión de riesgos supone un análisis crítico de futuro impacto del tratamiento para gestionar los problemas potenciales (amenazas) antes de que se conviertan en problemas reales.

**Gestión para los derechos y libertades de los sujetos de los datos**

El artículo 24 del RGPD establece que el responsable del tratamiento aplicará medidas técnicas y organizativas apropiadas. Incluir en un tratamiento un sistema de IA agéntica cambia la naturaleza del tratamiento y requiere un nuevo ciclo de gestión del riesgo.

**Regla de 2**

Una aproximación simplificada para fijar un umbral mínimo de garantías que jamás hay que traspasar. La Regla de 2 establece tres configuraciones que deben evitarse sin supervisión estricta o garantías extremas de integridad cuando intervienen la recolección automática de información incontrolada, el acceso a información sensible y la capacidad de realizar acciones automáticas de forma simultánea.

**Riesgo del tratamiento**

La gestión del riesgo se ha de realizar sobre el tratamiento en el que el sistema de agéntica es un medio con el objeto de proteger los derechos de los interesados.

**Efectos colaterales de los tratamientos**

El implementar tratamientos con técnicas novedosas puede originar efectos colaterales no deseados, como la inyección de prompts por propios clientes o personas usuarias.

**Evaluación de impacto para la protección de datos**

Los agentes de IA no implican necesariamente la obligación de realizar una EIPD en todos los casos, dependerá del tratamiento específico.

**Integración en la gestión de riesgo de la organización**

Las acciones para la mitigación del riesgo han de estar coordinadas con los demás aspectos de gestión de riesgos de la organización.

### **H. PROTECCIÓN DE DATOS DESDE EL DISEÑO Y POR DEFECTO**

El agente de IA deberá estar diseñado para recoger solo los datos estrictamente necesarios para el tratamiento al que da soporte, usarlos exclusivamente para el objetivo declarado, minimizar, aislar y proteger datos personales en cada paso del ciclo de vida. Es imprescindible la participación de DPDs y asesores en protección de datos debidamente cualificados.

### **I. TRANSFERENCIAS INTERNACIONALES**

Si por la inclusión de sistemas de AI agéntica se producen transferencias adicionales a un tercer país o a una organización internacional, se ha de asegurar que se realizan con las garantías del Capítulo V del RGPD.

## **VI. AMENAZAS**

La integración de agentes de IA introduce una nueva y ampliada superficie de ataque que va más allá del simple engaño a los modelos de IAG.

### **A. PROCEDENTES DEL TRATAMIENTO AUTORIZADO**

* **Falta de gobernanza y políticas en la organización:** No integrar la IA agéntica en la gobernanza y políticas de la entidad.  
* **Falta de madurez en el desarrollo:** Implementar soluciones utilizando metodologías y tecnologías no maduras.  
* **Falta de una política de acceso a los datos de la organización y del usuario:** Genera tratamiento excesivo, comunicación fuera de finalidades o exposición de datos personales.  
* **Falta de control del proceso de razonamiento:** Planificación errónea, incumplimiento del principio de minimización, exactitud y limitación.  
* **Desalineación:** El agente persigue metas que divergen de los objetivos de la organización o normativas.  
* **Realimentación en bucle y efectos burbuja:** Generación de ecosistemas cerrados que distorsionan decisiones al priorizar datos sesgados.  
* **Falta de control en el acceso a información externa:** Permite un scraping masivo o un tratamiento descontrolado de datos públicos.  
* **Exfiltración shadow-leak:** Filtración silenciosa y progresiva de información sensible a través de interacciones aparentemente legítimas.  
* **Desplazar toda la responsabilidad al usuario o a la supervisión humana:** Pretender suplir problemas subyacentes de diseño mediante supervisión humana.  
* **Falta de compartimentación de la memoria del agente:** Uso de memoria común para distintos tratamientos causando tratamientos excesivos.  
* **Falta de filtrado y saneamiento de información no estructurada y metadatos:** Exposición de datos ocultos en metadatos.  
* **Retención excesiva de datos:** Sin políticas de borrado en la memoria a largo plazo.  
* **Sesgo de automatización:** Confianza excesiva en las decisiones del sistema.  
* **Perfilado de los usuarios de la IA agéntica:** Creación de perfiles detallados del comportamiento de empleados.  
* **Disponibilidad y resiliencia:** Dependencia de servicios de Internet fuera de control de la organización.  
* **Acceso a la IA agéntica por usuarios no cualificados.**  
* **Compromisos en la cadena de suministro:** Vulnerabilidades heredadas de terceros.

### **B. PROCEDENTES DE TRATAMIENTOS NO AUTORIZADOS**

* **Inyección de prompts (Directa e Indirecta):** Instruir al modelo para comportarse de manera no prevista, incluyendo inyecciones ocultas en fuentes consultadas. Esto deriva en: Envenenamiento de memoria y RAG, Ataques de "Cero Clic", Exfiltración de datos mediante parámetros de URL, Secuestro de sesión y movimiento lateral, Ingeniería social dirigida a la IA, Ataques en pipeline largos, Confusión de contexto, Ataques diferidos, Escalada de privilegios mediante herramientas, Toma de control de la pantalla, Ataques de ransomware y borrado.  
* **Disponibilidad y resiliencia de servicios externos:** Suplantaciones o ataques de denegación de servicio (DoS) a las APIs externas.  
* **Acceso ilícito a la memoria agéntica:** Extracción de datos no autorizada de los registros del agente.

## **VII. MEDIDAS**

Existen múltiples medidas que permiten obtener los beneficios de incluir IA agéntica garantizando que el tratamiento es conforme al RGPD.

### **A. GOBERNANZA Y PROCESOS DE GESTIÓN**

La existencia de un marco de gobernanza de la información en la entidad es la medida más importante.

* **Aceptar la posibilidad de fallo:** Diseñar tratamientos asumiendo que ocurrirán errores para reaccionar apropiadamente.  
* **El Delegado de Protección de Datos:** Integrar a un DPD que conozca la normativa y la tecnología.  
* **Elementos básicos que hay que incorporar:** Asignar roles, prever efectos colaterales, determinar casos de uso, controlar el despliegue y establecer canales de información ágiles.

### **B. EVALUACIÓN CONTINUA DEL AGENTE BASADA EN EVIDENCIAS**

Automatizar la supervisión de los procesos con respecto al cumplimiento de políticas.

* **Establecimiento de criterios y métricas claras de funcionamiento.**  
* **Prácticas de "Golden testing":** Procedimientos para comparar el resultado actual de un sistema con un resultado de referencia.  
* **Contratos y otros vínculos legales:** Revisión dinámica de contratos de los múltiples servicios integrados.  
* **Aplicar el principio de precaución:** Enfoque incremental en la adopción de agentes.  
* **Explicabilidad:** Conseguirla mediante análisis de "caja blanca" y pruebas de "caja negra".  
* **Intervención humana:** Auditoría regular sobre la eficacia de la supervisión.

### **C. MINIMIZACIÓN DE DATOS**

Limitar el tratamiento de datos personales a lo estrictamente necesario.

* **Definición de políticas de acceso a la información:** Implementar el principio "need to know".  
* **Catálogo y catalogación de datos:** Gestionar activos de datos mediante metadatos.  
* **Catalogación de fuentes no estructuradas:** Etiquetado automatizado y extracción de datos específicos.  
* **Granularidad de la minimización:** Minimizar tanto a nivel de tratamiento como de operaciones de tratamiento.  
* **Filtrado de flujos de datos:** Analizar y limpiar los datos en tránsito entre acciones del agente.  
* **Shadow leaks:** Uso de herramientas DPL (data loss prevention) para minimizar la exposición del contexto interno.  
* **Seudonimización de las personas usuarias:** Uso de tokens de un solo uso.  
* **Control y perfilado de las personas usuarias:** Política estricta en la recolección de información de interacción del usuario.

### **D. CONTROL DE LA MEMORIA**

Realizado tanto sobre la memoria a corto plazo como sobre la memoria a largo plazo.

* **Gestión de memoria:** Acceder, catalogar y gestionar el contenido de la memoria.  
* **Compartimentación de la memoria:** Separar la memoria por tratamientos, casos o usuarios.  
* **Análisis y filtrado de la memoria de la persona usuaria:** Diferenciar entre la memoria de la organización y la memoria de la persona usuaria.  
* **No log policy selectivo:** Política de cero retención de datos a nivel de componente LLM.  
* **Establecimiento de plazos de retención estrictos.**  
* **Desactivación del almacenamiento en memoria:** Posibilidad de apagar la memoria persistente.  
* **Aplicar estrategias de higienización de la memoria:** Depuración automática de contenido sin uso u obsoleto.

### **E. AUTOMATIZACIÓN**

* **Decisión sobre el grado de autonomía:** Establecida por el responsable basándose en evidencias.  
* **Diseño eficaz y seguro de las cadenas de razonamiento:** Validado, incluyendo prevención de inyecciones mediante guardrails.  
* **Catálogo y listas blancas de servicios:** Identificando versiones y fiabilidad.  
* **Limitación de servicios accesibles.**  
* **Control en la ejecución de herramientas:** Guardrails en los parámetros de entrada y respuesta de APIs.  
* **Criterios y puntos de control para la intervención humana:** Acciones irreversibles o de alto impacto.  
* **Reversibilidad de las acciones de los agentes de IA.**  
* **Nivel de autonomía de acuerdo al tratamiento.**  
* **Supervision humana efectiva:** Evaluando la competencia, formación, independencia y recursos del supervisor.  
* **Rutas de escalamiento:** Monitorización en tiempo real para escalar situaciones de riesgo a un operador humano.  
* **Principio de los cuatro ojos:** Doble verificación para procesos de alto impacto.

### **F. CONTROL DEL AGENTE DESDE EL DISEÑO**

* **Documentación:** Registro exhaustivo de la arquitectura e incidencias.  
* **Profesionales cualificados:** Uso de personal formado en IA y normativa.  
* **Trazabilidad:** (Data Lineage) para conocer el ciclo de vida de los datos, facilitar el ejercicio de derechos y control de encargados.  
* **Test de verificación y validación:** Pruebas dinámicas y estáticas del código.  
* **Definir y controlar que los prompts siguen un procedimiento operativo estándar (SOP).**  
* **Mecanismos de repetibilidad:** Registrar parámetros y semillas para replicar decisiones.  
* **Gestión de identidad, autenticación, y privilegios:** Aplicar el principio de menor privilegio, mecanismos MFA, restringir la escalada de permisos.  
* **Control estricto sobre las actualizaciones:** Control de versiones con "roll-back".  
* **Sandboxing en desarrollo y explotación:** Entornos confinados para restringir el rango de acción de los agentes.  
* **Protocolos de detección de errores y planes de contingencia.**  
* **Control de flujo de extracción de datos:** Acciones expresas del usuario para envíos masivos.  
* **Cortacircuitos y límites duros de pasos (circuit breakers):** Interrupción automática ante bucles o anomalías.  
* **Controles de calibrado y alineación:** Evaluación en las fases intermedias de las cadenas de razonamiento.

### **G. GESTIÓN DEL CONSENTIMIENTO**

Implementar un mecanismo ágil para gestionar un ciclo de vida en el consentimiento, permitiendo granularidad respecto a categorías de datos, tratamientos y destinatarios (listas blancas/negras).

### **H. TRANSPARENCIA**

Ir más allá de los mínimos del RGPD para demostrar confianza, informando del flujo de datos en tiempo real, contexto utilizado en el resultado o acceso a registros de actividad.

### **I. ALFABETIZACIÓN**

Realizarse en tres niveles: nivel directivo (decisiones basadas en evidencias), nivel de responsables TIC y nivel de personas usuarias finales, con el DPD jugando un papel fundamental en este proceso formativo.

## **VIII. REFLEXIONES FINALES**

Los sistemas de IA han llegado para quedarse. Pretender ignorar su existencia supondría una pérdida de oportunidades estratégicas. Conocer esta tecnología es necesario para tomar decisiones racionales sobre su implementación basadas en evidencias.

Una implementación de la IA agéntica teniendo en cuenta la protección de datos desde el diseño permite definir tratamientos basados en agentes que incorporen técnicas de mejora de la privacidad (PETs) que ofrezcan garantías superiores a los tratamientos manuales. En sí misma, una IA agéntica puede ser una herramienta PET.

Para ello los DPD han de integrarse en las decisiones de diseño de los tratamientos y de los sistemas de agentes IA seleccionados para implementarlos. Esta tecnología está en plena evolución y requiere experiencia para analizar sus impactos, medidas y oportunidades en materia de protección de datos personales.

## **IX. REFERENCIAS**

* Reglamento (UE) 2016/679 (Reglamento General de Protección de Datos \- RGPD)  
* Directrices del Grupo de trabajo sobre Protección de Datos del Artículo 29 sobre decisiones individuales automatizadas y elaboración de perfiles (2017)  
* Agencia para la Ciberseguridad de la Unión Europea (ENISA). Towards a framework for policy development in cybersecurity (2018)  
* Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.  
* Agencia Española de Protección de Datos (AEPD). Adecuación al RGPD de tratamientos que incorporan Inteligencia Artificial (2020)  
* Agencia Española de Protección de Datos (AEPD). Gestión del riesgo y evaluación de impacto en tratamientos de datos personales (2021).  
* Comité Europeo de Protección de Datos. Directrices 07/2020 sobre los conceptos de responsable y encargado del tratamiento en el RGPD (2021)  
* Agencia Española de Protección de Datos (AEPD). Requisitos para Auditorías de Tratamientos que incluyan IA (2021)  
* Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models.  
* Agencia Española de Protección de Datos (AEPD). Evaluación de la intervención humana en las decisiones automatizadas (2024)  
* Agencia Española de Protección de Datos (AEPD). Introducción a LIINE4DU 1.0 (2024)  
* Future of Privacy Forum (FPF). Minding Mindful Machines: AI Agents and Data Protection Considerations (2024)  
* Anthropic. Model Context Protocol (MCP) Specification (2024)  
* Reglamento (UE) 2024/1689 de Inteligencia Artificial (RIA)  
* IBM. What are AI agents? (2025)  
* Park, T. (2024). Enhancing anomaly detection in financial markets with an LLM-based multi-agent framework.  
* Sapkota, R., et al. (2025/2026). AI Agents vs. Agentic AI.  
* OWASP Foundation. Agentic AI-threats and mitigations (2025)  
* Feng et al. Levels of Autonomy for AI Agents (2025)  
* Infocomm Media Development Authority. Model AI governance framework for agentic AI Singapur (2026)