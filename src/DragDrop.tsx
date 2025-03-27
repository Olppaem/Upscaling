import React, { useState,useEffect,useCallback } from "react";
import { FileUploader } from "react-drag-drop-files";
import { useDropzone } from 'react-dropzone';

export {};

const fileTypes: string[] = ["jpg", "jpeg", "png", "gif", "mp3"];

interface DragDropProps {
  title: string;
  files : File[];
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
  hideFiles : boolean;
  validateFileName: (fileName: string) => boolean; 
}

const DragDrop: React.FC<DragDropProps> = ({ title, files, setFiles,validateFileName }) => { 
  const [localFiles, setLocalFiles] = useState<File[]>([]);


  useEffect(() => {
    setFiles(localFiles);
  }, [localFiles, setFiles]);

  const handleChange = (selectedFiles: File[] | FileList) => {
    let fileArray: File[] = [];
    if (Array.isArray(selectedFiles)) {
      fileArray = selectedFiles as File[]; // 타입 단언을 사용하여 selectedFiles를 File[]로 변환
    } else {
      // selectedFiles가 FileList인 경우
      fileArray = Array.from(selectedFiles) as File[]; // FileList를 배열로 변환
    }

    //파일 이름 검사
    const validFiles = fileArray.filter(file => validateFileName(file.name));
    const invalidFiles = fileArray.filter(file => !validateFileName(file.name));

    if (invalidFiles.length > 0) {
      const invalidFileNames = invalidFiles.map(file => file.name).join(', ');
      alert(`다음 파일은 필터링되었습니다: ${invalidFileNames}`);
    }

    setLocalFiles(prevFiles => [...prevFiles, ...validFiles]);
  };

  const handleDelete = (indexToDelete: number) => {
    setLocalFiles((prevFiles) => 
      prevFiles.filter((_, index) => index !== indexToDelete));
  };



  return (
    <div style={{ width: '100vw', height: '100vw', margin: '0', padding: '0', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <h3>{title}</h3>
      <div style={{padding : "16px", width: '40vw', height: '60vh', border: '2px dashed #CCCCCC', borderRadius: '10px', display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column' }}>
        <FileUploader
          handleChange={handleChange}
          name="file"
          types={fileTypes}
          multiple={true}
          maxSize={10}
        >
          <p style={{ fontSize: '24px', color: '#666' }}>파일을 여기에 드래그 앤 드롭하세요</p>
          <p style={{ fontSize: '18px', color: '#999' }}>또는 클릭하여 파일을 선택하세요</p>
        </FileUploader>
      
      {localFiles.length > 0 && (
        <div style={{ width: '40vw', marginTop: '20px' }}>
          <h4>업로드된 파일 목록:</h4>
          <ul style={{ listStyleType: 'none', padding: 0 }}>
            {localFiles.map((file, index) => (
              <li key={index} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <span>{file.name}</span>
                <button onClick={() => handleDelete(index)} style={{ background: 'none', border: 'none', color: 'black', cursor: 'pointer' }}>
                  삭제
                </button>
              </li>
            ))}
          </ul>
        </div>

      )}
      </div>
    </div>
  );
};

export default DragDrop;