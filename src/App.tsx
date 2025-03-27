import logo from './logo.svg';
import './App.css';
import { Tab, TabList,Tabs,TabPanel } from "react-tabs";
import 'react-tabs/style/react-tabs.css';
import DragDrop from './DragDrop';
import axios from 'axios';
import React, {useState} from 'react';

export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [upscaledImageUrl, setUpscaledImageUrl] = useState<string | null>(null);
  const [upscaleFiles, setUpscaleFiles] = useState<File[]>([]);
  const [compressFiles, setCompressFiles] = useState<File[]>([]);
  const [hideUpscaleFiles, setHideUpscaleFiles] = useState(false);
  const [audioFiles, setAudioFiles] = useState<File[]>([]);
  const [failedFiles, setFailedFiles] = useState<string[]>([]);


  const handleConfirm = async () => {
    setIsLoading(true);
    setHideUpscaleFiles(true);

    setFailedFiles([]); // Failed files 초기화

    const processFiles = async (files: File[], endpoint: string) => {
      const failedFilesList: string[] = [];
      
      const promises = files.map(async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        try {
          const response = await axios.post(endpoint, formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            withCredentials: true
          });

          if (response.data.status === 'success') {
            setUpscaledImageUrl(`http://localhost:8000/${response.data.image_path}`);
          } else {
            failedFilesList.push(response.data.filename || file.name);
          }
        } catch (error) {
          console.error("처리 중 오류 발생:", error);
          failedFilesList.push(file.name);
        }
      });

      await Promise.all(promises);
      return failedFilesList;
    }

    const upscaleFailedFiles = await processFiles(upscaleFiles, 'http://localhost:8000/upscale');
    const compressFailedFiles = await processFiles(compressFiles, 'http://localhost:8000/compress');
    const audioFailedFiles = await processFiles(audioFiles, 'http://localhost:8000/normalize_audio');

    const allFailedFiles = [...upscaleFailedFiles, ...compressFailedFiles, ...audioFailedFiles];
    setFailedFiles(allFailedFiles);

    setIsLoading(false);
    setHideUpscaleFiles(false);

    if (allFailedFiles.length > 0) {
      alert(`업스케일링에 실패한 파일: ${allFailedFiles.join(', ')}`);
    } else {
      alert("모든 파일 처리가 완료되었습니다!");
    }
  }
  
  const validateFileName = (fileName: string): boolean => {
    const pattern = /^[\w]+_[\w]+\.png$/;
    return pattern.test(fileName);
  };

  const UpscalerPageTabs =()=>{
    return(
        <Tabs>
            <TabList>
                <Tab>탐정 엔버</Tab>
                <Tab>봉순</Tab>
                <Tab>love is near blind</Tab>
                <Tab>단어 이미지</Tab>
                <Tab>공통 음성</Tab>
            </TabList>
            <TabPanel>
                <h2>탐정 엔버</h2>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <DragDrop title="upscale" files = {upscaleFiles} setFiles={setUpscaleFiles} hideFiles={hideUpscaleFiles} validateFileName={validateFileName}/>
                    <DragDrop title="not upscale" files = {compressFiles}  setFiles={setCompressFiles} hideFiles={false}   validateFileName={validateFileName}/>
                </div>
            </TabPanel>
            <TabPanel>
                <h2>봉순</h2>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <DragDrop title="upscale"files = {upscaleFiles}  setFiles={setUpscaleFiles} hideFiles={hideUpscaleFiles}validateFileName={validateFileName}/>
                    <DragDrop title="not upscale" files = {compressFiles} setFiles={setCompressFiles} hideFiles={false} validateFileName={validateFileName}/>
                </div>
            </TabPanel>
            <TabPanel>
                <h2>love is near blind</h2>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <DragDrop title="upscale" files = {upscaleFiles}  setFiles={setUpscaleFiles} hideFiles={hideUpscaleFiles} validateFileName={validateFileName}/>
                    <DragDrop title="not upscale" files = {compressFiles}   setFiles={setCompressFiles}  hideFiles={false} validateFileName={validateFileName}/>
                </div>
            </TabPanel>
            <TabPanel>
              <h2>단어 이미지</h2>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <DragDrop title="upscale"files = {upscaleFiles}  setFiles={setUpscaleFiles} hideFiles={hideUpscaleFiles} validateFileName={validateFileName}/>
                    <DragDrop title="not upscale" files = {compressFiles} setFiles={setCompressFiles} hideFiles={false} validateFileName={validateFileName}/>
              </div>
            </TabPanel>
            <TabPanel>
              <h2>공통 음성</h2>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <DragDrop title="normalize audio"files = {audioFiles}  setFiles={setAudioFiles} hideFiles={false} validateFileName={validateFileName}/>
              </div>
            </TabPanel>
        </Tabs>
    );
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <div style={{ flex: 1 }}>
        <h1>Upscaler Page</h1>
        {UpscalerPageTabs()}
      </div>
      <div style={{ display: 'flex', justifyContent: 'center', padding: '20px',position: 'fixed', bottom: 0, width: '100%' }}>
        <button 
          onClick={handleConfirm}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#7e7e7e',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          {isLoading? '처리 중...':'Upscale'}
        </button>
      </div>
    </div>
  );
}

