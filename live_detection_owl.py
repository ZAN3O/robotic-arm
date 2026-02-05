import cv2
import torch
import time
import argparse
import numpy as np
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from vision import CameraCapture

def parse_args():
    parser = argparse.ArgumentParser(description="Live Object Detection with Google OWL-ViT (Hugging Face)")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--labels", type=str, 
        default="pen, notebook, ruler, eraser, glue stick, scissors, glasses",
        help="Comma-separated labels (ENGLISH recommended for OWL-ViT)"
    )
    parser.add_argument("--score-threshold", type=float, default=0.1)
    return parser.parse_args()

def main():
    args = parse_args()
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")
    
    print(f"🔄 Loading model {args.model}...")
    processor = OwlViTProcessor.from_pretrained(args.model)
    model = OwlViTForObjectDetection.from_pretrained(args.model).to(device)
    model.eval()
    
    # Pre-tokenize text queries (moved outside loop)
    text_queries = labels
    inputs_text = processor(text=text_queries, return_tensors="pt").to(device)
    input_ids = inputs_text["input_ids"]
    attention_mask = inputs_text["attention_mask"]
    
    cap = CameraCapture(args.camera_id, 1280, 720)
    if not cap.open():
        return

    print("✅ Starting inference...")
    cv2.namedWindow("OWL-ViT", cv2.WINDOW_NORMAL)
    
    t0 = time.perf_counter()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Preprocess image only
            t_pre_start = time.perf_counter()
            inputs_img = processor(images=pil_img, return_tensors="pt").to(device)
            pixel_values = inputs_img["pixel_values"]
            t_pre_end = time.perf_counter()
            
            t_inf_start = time.perf_counter()
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            t_inf_end = time.perf_counter()
            
            # Post-process
            target_sizes = torch.Tensor([pil_img.size[::-1]]).to(device)
            results = processor.image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=args.score_threshold)[0]
            t_post_end = time.perf_counter()
            
            pre_ms = (t_pre_end - t_pre_start) * 1000
            inf_ms = (t_inf_end - t_inf_start) * 1000
            post_ms = (t_post_end - t_inf_end) * 1000
            
            # Draw
            font = cv2.FONT_HERSHEY_SIMPLEX
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                x1, y1, x2, y2 = map(int, box)
                label_text = f"{text_queries[label]}: {round(score.item(), 2)}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

            # FPS
            frame_count += 1
            if frame_count % 5 == 0:
                dt = time.perf_counter() - t0
                fps = 5 / dt
                t0 = time.perf_counter()

            cv2.putText(frame, f"FPS: {fps:.1f} | Pre:{pre_ms:.0f} Inf:{inf_ms:.0f} Post:{post_ms:.0f} (ms)", (10, 30), font, 0.7, (0, 0, 255), 2)
            cv2.imshow("OWL-ViT", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
