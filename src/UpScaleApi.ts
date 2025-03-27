import axios, {AxiosInstance} from "axios";
import {  ExtensionRequest, ImageInfo, UpscaleRequest } from "./UpScaleModels";

class UpScaleApi{
    private static instance: UpScaleApi;
    private axiosInstance: AxiosInstance;

    private constructor() {
      this.axiosInstance = axios.create({
        baseURL: " ", // 실제 사용할 API의 주소
        withCredentials: true, // 필요에 따라 설정
      });
    }
    public static getInstance(): UpScaleApi {
      if (!UpScaleApi.instance) {
        UpScaleApi.instance = new UpScaleApi();
      }
      return UpScaleApi.instance;
    }

    // 이미지 정보 가져오기
    async FetchImagesApi(): Promise<ImageInfo[]> {
      const endpoint = `${this.axiosInstance.defaults.baseURL}/images/`;
        try {
            const response = await this.axiosInstance.get(endpoint);
            console.log(response.data);
            return response.data;
        } catch (error) {
            console.error(error);
            throw error;
        }
    }
    //이미지 업스케일러
    async UpScaleImageApi(req: UpscaleRequest): Promise<ImageInfo>{//요청, 반환 모델
      const endpoint = `${this.axiosInstance.defaults.baseURL}/images/upscale`;
      try {
        const response = await this.axiosInstance.post<ImageInfo>(endpoint, req);
        console.log(response.data);
        alert("UpScale Success");
        return response.data;
      } catch (error) {
        console.error(error);
        throw error;
      }
    }
    //webp변환
    async changeExtensionApi(req: ExtensionRequest):Promise<ImageInfo>{
      const endpoint = `${this.axiosInstance.defaults.baseURL}/images/change`;
      try {
        const response = await this.axiosInstance.post<ImageInfo>(endpoint, req);
        console.log(response.data);
        return response.data;
      } catch (error) {
        console.error(error);
        throw error;
      }
    }

    
    
    
  }
const UpScaleApiInstance = UpScaleApi.getInstance();
export default UpScaleApi;
