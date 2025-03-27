//이미지 정보 가져오기
export interface ImageInfo {
  index: number;
  series: string;
  imageUrl: string;
  extension: string;
  fileSize: number;
}
//이미지 업스케일러
export interface UpscaleRequest {
    imageUrl: string;
  }
//webp 변환
export interface ExtensionRequest {
  imageUrl: string;
  extension: string;
}
//파일압축
export interface CompressRequest {
    imageUrl: string;
    extension: string;
}
//클라우드에 저장
export interface SaveToCloudRequest {
  index: number;
  series: string;
  imageUrl: string;
  extension: string;
  fileSize: number;
}