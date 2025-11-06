# VISOR Architecture Diagrams

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph "Frontend"
        A[Web Browser] --> B[Camera Feed]
        B --> C[Frame Capture]
        C --> D[UI Controls]
        D --> E[Text-to-Speech]
    end
    
    subgraph "Backend API"
        F[FastAPI Server] --> G[/analyze endpoint]
        G --> H[Image Processing]
    end
    
    subgraph "Vision Models"
        H --> I[YOLOv8 Detection]
        H --> J[BLIP Captioning]
        H --> K[BLIP VQA]
    end
    
    subgraph "Fusion Layer"
        I --> L[Detection Summarization]
        J --> L
        L --> M[FLAN-T5/Gemini Reasoner]
        M --> N[Fused Narrative]
    end
    
    C --> F
    N --> G
    G --> A
    A --> E
    
    style A fill:#7cc4ff
    style F fill:#233055,color:#e6eef8
    style M fill:#64e1a7
    style N fill:#64e1a7
```

## 2. Multimodal Fusion Pipeline

```mermaid
flowchart LR
    A[Input Image] --> B[YOLO Detector]
    A --> C[BLIP Captioner]
    A --> D[BLIP VQA]
    
    B --> E["Detections:<br/>classes, boxes,<br/>confidences"]
    C --> F["Caption:<br/>Natural language<br/>description"]
    
    E --> G[Fusion Layer]
    F --> G
    
    G --> H["Prompt Construction:<br/>Instruction + Caption +<br/>Detections"]
    H --> I[Language Model<br/>FLAN-T5/Gemini]
    I --> J[Fused Narrative]
    
    D --> K[VQA Answer]
    
    J --> L[Output]
    K --> L
    
    style A fill:#7cc4ff
    style G fill:#64e1a7
    style I fill:#64e1a7
    style J fill:#64e1a7
    style L fill:#7cc4ff
```

## 3. Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant FastAPI
    participant YOLO
    participant BLIP
    participant Reasoner
    
    User->>Browser: Start Camera
    Browser->>Browser: Capture Frame (1.5s interval)
    Browser->>FastAPI: POST /analyze (image + question)
    
    FastAPI->>YOLO: detect(image)
    YOLO-->>FastAPI: detections [boxes, classes, conf]
    
    FastAPI->>BLIP: caption(image)
    BLIP-->>FastAPI: caption text
    
    FastAPI->>BLIP: vqa(image, question)
    BLIP-->>FastAPI: vqa_answer
    
    FastAPI->>Reasoner: generate_narrative(caption, detections)
    Reasoner-->>FastAPI: fused narrative
    
    FastAPI-->>Browser: JSON {narrative, caption, vqa_answer, detections}
    Browser->>Browser: Display results
    Browser->>Browser: Speak narrative (TTS)
    Browser-->>User: Audio output
```

## 4. Frontend-Backend Interaction

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[index.html] --> B[app.js]
        B --> C[Camera Capture]
        B --> D[Voice Input]
        B --> E[Speech Synthesis]
        B --> F[UI Controls]
    end
    
    subgraph "Communication"
        C --> G[POST /analyze]
        D --> G
        G --> H[JSON Response]
    end
    
    subgraph "Backend Layer"
        H --> I[backend.py]
        I --> J[analyze_image]
        J --> K[detect]
        J --> L[caption]
        J --> M[vqa]
        J --> N[generate_narrative]
    end
    
    H --> B
    B --> E
    
    style A fill:#7cc4ff
    style B fill:#7cc4ff
    style I fill:#233055,color:#e6eef8
    style N fill:#64e1a7
```

## 5. Model Pipeline & Evaluation

```mermaid
graph LR
    subgraph "Input"
        A[Image] --> B[YOLOv8n]
        A --> C[BLIP-base]
    end
    
    subgraph "Individual Models"
        B --> D["Output 1:<br/>Detections<br/>mAP50: 0.605"]
        C --> E["Output 2:<br/>Caption<br/>BLEU: varies"]
    end
    
    subgraph "Fusion"
        D --> F[Fusion Layer]
        E --> F
        F --> G[Reasoner<br/>FLAN-T5/Gemini]
        G --> H["Output 3:<br/>Fused Narrative<br/>Object Coverage: 54.08%"]
    end
    
    subgraph "Evaluation"
        D --> I[YOLO Metrics:<br/>Precision, Recall,<br/>mAP50, mAP50-95]
        E --> J[Caption Metrics:<br/>BLEU, ROUGE-L,<br/>METEOR]
        H --> K[Fusion Metrics:<br/>Object Coverage +65.1%<br/>ROUGE-L: 0.7484<br/>METEOR: 0.7374]
    end
    
    style A fill:#7cc4ff
    style F fill:#64e1a7
    style G fill:#64e1a7
    style H fill:#64e1a7
    style K fill:#f6f8fc
```

## 6. Complete System Workflow

```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> LoadModels: Startup
    LoadModels --> Ready: Models Loaded
    
    Ready --> CameraStart: User clicks Start
    CameraStart --> Capturing: Camera Active
    
    Capturing --> ProcessFrame: Every 1.5s
    ProcessFrame --> YOLODetection: YOLO Model
    ProcessFrame --> BLIPCaption: BLIP Model
    ProcessFrame --> BLIPVQA: If VQA enabled
    
    YOLODetection --> Fusion: Detections ready
    BLIPCaption --> Fusion: Caption ready
    BLIPVQA --> Fusion: VQA ready
    
    Fusion --> Reasoner: Generate narrative
    Reasoner --> OutputReady: Narrative generated
    
    OutputReady --> Display: Update UI
    Display --> TTS: Speak if enabled
    TTS --> Capturing: Continue loop
    
    Capturing --> CameraStop: User clicks Stop
    CameraStop --> Ready: Return to idle
    
    Ready --> [*]: Shutdown
    
    note right of Fusion
        Late Fusion:
        Combines YOLO + BLIP
        via Language Model
    end note
```

## Usage

These diagrams can be:
- Embedded in documentation (GitHub README, project docs)
- Exported as images using Mermaid Live Editor (https://mermaid.live)
- Included in presentations
- Added to the fusion report PDF (converted to images)

## Diagram Descriptions

1. **System Architecture**: Overall system components and their relationships
2. **Multimodal Fusion Pipeline**: Detailed fusion process showing how models combine
3. **Data Flow**: Sequence diagram showing request/response flow
4. **Frontend-Backend Interaction**: Component interactions across layers
5. **Model Pipeline & Evaluation**: Models with their evaluation metrics
6. **Complete System Workflow**: State machine showing system states and transitions

