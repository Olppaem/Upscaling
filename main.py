import replicate 
from webptools import cwebp
from PIL import Image
import requests
import os
from dotenv import load_dotenv
import asyncio
from webptools import cwebp
from io import BytesIO  
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import aiohttp
import aiofiles
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.playback import play
from datetime import datetime
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱의 주소를 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#ffmpeg 경로 설정
ffmpeg_path = r"C:/Users/user/ffmpeg-2024-07-22-git-172da370e7-full_build/bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

load_dotenv()
api_key = os.getenv('REPLICATE_API_TOKEN')


class EsrganClient:
    def __init__(self):
        self.scale_factor = 3
        self.face_enhance = False
        self.model = "nightmareai/real-esrgan:350d32041630ffbe63c8352783a26d94126809164e54085352f8326e53999085"
        self.api_key = api_key
  
    async def upscale(self, image_path):
        try : 
            start_time = datetime.now()
            print(f"Upscaling started at {start_time} for {image_path}")

            input_data = {
                "image": open(image_path, "rb"),
                "scale": self.scale_factor,
                "face_enhance": self.face_enhance
            }
            output = await asyncio.to_thread(replicate.run,self.model,input=input_data)
            
            end_time = datetime.now()
            print(f"Upscaling completed at {end_time} for {image_path}")
            
            print(f"Upscale output: {output}")
            return output
        except Exception as e:
            print(e)
    

    async def get_image_from_url(self, image_url):
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return Image.open(BytesIO(image_data))
                else:
                    print(f"Error fetching image from URL: {response.status}")
                    return None

    async def compress_webp(self, image, output_path, quality=75, method=6):
        await asyncio.to_thread(image.save,output_path, 'WEBP', quality=quality, method=method)
        print(f"Compression completed for {output_path}")

async def process_single_image(file: UploadFile):
    print("start")
    file_path = f"temp_{file.filename}"
    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        esrgan_client=EsrganClient()

    
        # 이미지 업스케일
        upscaled_image_url = await esrgan_client.upscale(file_path)
        
        if upscaled_image_url:
            # 업스케일된 이미지 가져오기
            upscaled_image = await esrgan_client.get_image_from_url(upscaled_image_url)
            
            if upscaled_image:
                # webp로 압축 및 저장
                compress_output_path = f'upscaled_{file.filename}.webp'
                await esrgan_client.compress_webp(upscaled_image, compress_output_path)
                
                print(f"Processing completed for {file.filename}")
                return {"status": "success", "image_path": compress_output_path}
            raise Exception(f"업스케일링 실패: {file.filename}")

        print(f"Processing failed for {file.filename}")
        return {"status": "error", "message": f"업스케일링 실패: {file.filename}"}
    
    except Exception as e:
        print(f"Error processing {file.filename}: {str(e)}")
        return {"status": "error", "message": str(e), "filename": file.filename}


    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    return await process_single_image(file)

@app.post("/upscale_multiple")
async def upscale_multiple_images(files: list[UploadFile] = File(...)):
    tasks = [process_single_image(file) for file in files]
    results =await asyncio.gather(*tasks)

    successful_files = []
    error_files = []

    for result in results:
        if isinstance(result, dict) and result["status"] == "success":
            successful_files.append(result)
        else:
            if isinstance(result, dict):
                error_files.append(result["filename"])
            elif isinstance(result, Exception):
                error_files.append(str(result))

    response_data = {
        "successful_files": successful_files,
        "error_files": error_files
    }

    if error_files:
        return JSONResponse(
            status_code=207,
            content=response_data
        )
    else:
        return JSONResponse(content=response_data)


@app.post("/compress")
async def compress_image(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        image = Image.open(file_path)
        compress_output_path = f'compressed_{file.filename}.webp'
        await asyncio.to_thread(image.save, compress_output_path, 'WEBP', quality=75, method=6)

        print(f"Compression completed for{file.filename}")
        # 처리된 이미지의 경로 반환
        return {"status": "success", "image_path": compress_output_path}

    finally:
        # 임시 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/normalize_audio")
async def normalize_audio_endpoint(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        with open(file_path, "rb") as f:
            audio_bytes_io = BytesIO(f.read())

        normalized_audio_bytes_io = await asyncio.to_thread(normalize_audio, audio_bytes_io)
        
        normalized_audio_path = f"normalized_{file.filename}"
        async with aiofiles.open(normalized_audio_path, 'wb') as out_file:
            await out_file.write(normalized_audio_bytes_io.read())

        print(f"Normalization completed for {file.filename}")
        return {"status": "success", "audio_path": normalized_audio_path}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def normalize_audio(audio_bytes_io: BytesIO) -> BytesIO:
    audio_bytes_io.seek(0)
    audio = AudioSegment.from_file(audio_bytes_io)

    #음성 길이 조정
    max_length_ms = 60 * 1000  
    if len(audio) > max_length_ms:
        audio = audio[:max_length_ms]
    
    #페이드 아웃 
    fade_duration_ms = 15 * 1000  # 15초
    if len(audio) > 45 * 1000:
        fade_start = len(audio) - fade_duration_ms
        audio = audio.fade_out(fade_duration_ms)

    #정규화
    normalized_audio = normalize(audio, headroom=4.0)

    #최종 오디오 저장
    normalized_audio_bytes_io = BytesIO()
    normalized_audio.export(normalized_audio_bytes_io, format="mp3")
    normalized_audio_bytes_io.seek(0)
    return normalized_audio_bytes_io



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



    # 1. PNG Sample Image 잡고
    # api_key env 에 저장
    # 2. Run Upscale (몇 초 걸리는지) 약5초 
    #esrgan_client.upscale() #output 확인
    # 3. #2의 output 확인하고 png면 webp로. 약 5초
    # webp면 그냥 그대로  
    # 4. webp 압축 
